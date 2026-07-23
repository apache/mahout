//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Pure, allocation-free memory estimation for pipeline configurations.
//!
//! [`estimate_memory`] answers "how much memory will this config need?" from the config alone —
//! no allocation, no CUDA calls, no device queries — so callers can fail fast before the first
//! batch instead of discovering an overflow mid-run.

use crate::error::{MahoutError, Result};
use crate::types::{Dtype, Encoding};

/// Predicted memory footprint of a pipeline configuration, in bytes.
///
/// Produced by [`estimate_memory`]. Both fields are upper-bound estimates derived from config
/// arithmetic only; nothing here reflects an actual allocation. See [`estimate_memory`] for the
/// formulas behind each field and a worked example.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// Host-side prefetch pool: `prefetch_depth` batches of raw (unencoded) input.
    pub cpu_prefetch_bytes: u64,
    /// Device-side state-vector buffer, including a double-buffering allowance.
    pub gpu_state_bytes: u64,
}

impl MemoryEstimate {
    /// Combined host + device footprint in bytes.
    ///
    /// Saturates instead of overflowing: values produced by [`estimate_memory`] are checked to sum
    /// without overflow, but the fields are public and may be set directly.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.cpu_prefetch_bytes.saturating_add(self.gpu_state_bytes)
    }
}

/// Estimate the CPU prefetch-pool and GPU state-buffer footprint of a pipeline configuration.
///
/// Pure function: performs no allocation, touches no device, and never panics — every product is
/// computed with checked arithmetic and an overflow becomes [`MahoutError::InvalidInput`] naming
/// the parameter responsible.
///
/// # Formulas
///
/// Let `n = num_qubits`, `bpe = dtype.bytes()` (4 for f32, 8 for f64), and
/// `complex_bytes = 2 * bpe` (real + imaginary component).
///
/// | Field                | Formula                                             |
/// |----------------------|-----------------------------------------------------|
/// | `cpu_prefetch_bytes` | `prefetch_depth * batch_size * sample_len * bpe`     |
/// | `gpu_state_bytes`    | `2 * batch_size * state_len * complex_bytes`         |
/// | `total()`            | `cpu_prefetch_bytes + gpu_state_bytes`              |
///
/// `sample_len` is the per-sample **input** length and depends on the encoding
/// ([`Encoding::vector_len`]): `2^n` for amplitude, `n` for angle/iqp-z/phase,
/// `n + n*(n-1)/2` for iqp, `1` for basis. `state_len` is always `2^n` — every encoding produces a
/// full state vector on the device regardless of how small its input is.
///
/// # `prefetch_depth`
///
/// Pass an **already-resolved** depth: this function does not replicate the pipeline's
/// auto-compute logic (`PipelineConfig::normalize` / `compute_optimal_prefetch_depth`). `0` is that
/// logic's sentinel, so passing it estimates a zero-byte pool rather than what the pipeline would
/// actually allocate — run the config through `normalize` first.
///
/// # dtype fallback
///
/// A `Float32` request for an encoding without an f32 batch path
/// ([`Encoding::supports_f32`]) is downgraded to `Float64` before estimating, mirroring
/// `PipelineConfig::normalize` — otherwise the estimate would report half the memory the pipeline
/// actually uses.
///
/// # Modeling caveats
///
/// 1. **The leading `2×` on `gpu_state_bytes` is a double-buffering allowance (state vector +
///    recycle buffer), not the complex real/imaginary factor** — the real/imag pair is already
///    inside `complex_bytes`. The batch path (`GpuStateVector::new_batch`) allocates a single
///    `batch_size × state_len × complex_bytes` buffer, so the `2×` is a deliberate conservative
///    upper bound.
/// 2. **`cpu_prefetch_bytes` covers exactly `prefetch_depth` batches.** The true instantaneous peak
///    can reach `prefetch_depth + 2` batches (one being produced and one being consumed alongside a
///    full channel); this `+2` is **not** folded into the returned value, so the number stays
///    aligned with the formula above. Budget headroom accordingly.
/// 3. **The host-side streaming chunk buffer (Parquet reader) is excluded.** This estimate covers
///    only the prefetch pool and GPU state buffer; folding in the reader's chunk buffer needs
///    reader config beyond this signature and is tracked as a follow-up. B2 must account for it
///    separately for streaming sources.
///
/// # Errors
///
/// Returns [`MahoutError::InvalidInput`] when `num_qubits` is too large for `2^n` to be
/// representable, or when any product overflows `u64`.
///
/// # Examples
///
/// 16-qubit amplitude, f32, batch 64, prefetch depth 16:
///
/// ```
/// use qdp_core::{Dtype, Encoding, estimate_memory};
///
/// let est = estimate_memory(Encoding::Amplitude, 16, 64, Dtype::Float32, 16).unwrap();
/// // cpu: 16 * 64 * 65536 * 4 bytes
/// assert_eq!(est.cpu_prefetch_bytes, 256 * 1024 * 1024);
/// // gpu: 2 * 64 * 65536 * 8 bytes
/// assert_eq!(est.gpu_state_bytes, 64 * 1024 * 1024);
/// assert_eq!(est.total(), 320 * 1024 * 1024);
/// ```
pub fn estimate_memory(
    encoding: Encoding,
    num_qubits: u32,
    batch_size: usize,
    dtype: Dtype,
    prefetch_depth: usize,
) -> Result<MemoryEstimate> {
    // Mirror PipelineConfig::normalize: f32 is only kept for encodings with a real f32 batch path.
    let dtype = if matches!(dtype, Dtype::Float32) && !encoding.supports_f32() {
        Dtype::Float64
    } else {
        dtype
    };

    // Device side always holds a full 2^n state vector per sample, whatever the input length is.
    // Compute it with checked_shl rather than Encoding::vector_len, which would panic on `1 << n`
    // for large n.
    let state_len = 1u64.checked_shl(num_qubits).ok_or_else(|| {
        MahoutError::InvalidInput(format!(
            "num_qubits too large to estimate: {num_qubits} (2^n state vector is not representable)"
        ))
    })?;

    let sample_len = match encoding {
        Encoding::Amplitude => state_len,
        _ => encoding.vector_len(num_qubits) as u64,
    };

    let bytes_per_elem = dtype.bytes() as u64;
    let complex_bytes = 2 * bytes_per_elem;
    let batch = batch_size as u64;
    let depth = prefetch_depth as u64;

    let cpu_prefetch_bytes = depth
        .checked_mul(batch)
        .and_then(|v| v.checked_mul(sample_len))
        .and_then(|v| v.checked_mul(bytes_per_elem))
        .ok_or_else(|| overflow_err("CPU prefetch pool", num_qubits, batch_size, prefetch_depth))?;

    let gpu_state_bytes = 2u64
        .checked_mul(batch)
        .and_then(|v| v.checked_mul(state_len))
        .and_then(|v| v.checked_mul(complex_bytes))
        .ok_or_else(|| overflow_err("GPU state buffer", num_qubits, batch_size, prefetch_depth))?;

    // total() adds the two; make sure that cannot overflow either.
    cpu_prefetch_bytes
        .checked_add(gpu_state_bytes)
        .ok_or_else(|| overflow_err("total footprint", num_qubits, batch_size, prefetch_depth))?;

    Ok(MemoryEstimate {
        cpu_prefetch_bytes,
        gpu_state_bytes,
    })
}

fn overflow_err(what: &str, num_qubits: u32, batch_size: usize, depth: usize) -> MahoutError {
    MahoutError::InvalidInput(format!(
        "{what} size overflows u64 for num_qubits={num_qubits}, batch_size={batch_size}, \
         prefetch_depth={depth}; reduce num_qubits or batch_size"
    ))
}
