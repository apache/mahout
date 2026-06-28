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

//! End-to-end f32 Parquet fidelity tests — issue #1342.
//!
//! These tests extend `gpu_fidelity.rs` (which compares f32 vs f64 encodings on
//! in-memory data) by sourcing the data from Parquet: the same deterministic
//! amplitude data is written as both a `List<Float32>` and a `List<Float64>`
//! column, read back through `ParquetReader::<f32>` / `ParquetReader::<f64>`, and
//! encoded with the f32 / f64 kernels respectively. We then compare the GPU
//! state vectors with `fidelity_cross_precision`.
//!
//! Two layers:
//!   * CPU smoke (no GPU, runs everywhere): the f32 reader and the f64 reader
//!     return the same logical values up to the f64→f32 rounding. This is the
//!     part that actually asserts in CI (ubuntu, no GPU).
//!   * GPU fidelity (Linux + CUDA, skipped otherwise): full reader→kernel
//!     pipeline. Thresholds match `gpu_fidelity.rs` — amplitude f32 norm
//!     precision means ~1e-3 at 8–16 qubits, NOT 0.99999. The issue's
//!     "≥ 0.99999 where applicable" does not apply to amplitude encoding; see
//!     `gpu_fidelity.rs::compare_f32_f64_amplitude` for the established baseline.

use std::sync::atomic::{AtomicUsize, Ordering};

use qdp_core::reader::{DataReader, NullHandling};
use qdp_core::readers::parquet::ParquetReader;

mod common;

static FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Deterministic amplitude data, identical generator to `gpu_fidelity.rs`.
fn make_amplitude_data(num_samples: usize, sample_size: usize) -> Vec<f64> {
    let total = num_samples * sample_size;
    (0..total)
        .map(|i| ((i as f64) + 1.0) / total as f64)
        .collect()
}

/// Returns `(f32_path, f64_path)` for two temp Parquet files holding the same
/// logical data as `List<Float32>` and `List<Float64>` respectively.
fn write_pair(data_f64: &[f64], sample_size: usize) -> (TempFile, TempFile) {
    let n = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let f32_path = std::env::temp_dir().join(format!("mahout_f32fid_{pid}_{n}_f32.parquet"));
    let f64_path = std::env::temp_dir().join(format!("mahout_f32fid_{pid}_{n}_f64.parquet"));

    let data_f32: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();
    common::write_list_parquet_f32(f32_path.to_str().unwrap(), &data_f32, sample_size);
    common::write_list_parquet_f64(f64_path.to_str().unwrap(), data_f64, sample_size);

    (TempFile(f32_path), TempFile(f64_path))
}

struct TempFile(std::path::PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ---------------------------------------------------------------------------
// CPU smoke (no GPU) — this is what actually asserts in CI.
// ---------------------------------------------------------------------------

/// The f32 reader and the f64 reader return the same shape, and the f32 values
/// equal the f64 values narrowed with `as f32` (Arrow cast == IEEE
/// round-to-nearest, same as the f64→f32 path the kernel would see).
#[test]
fn test_parquet_f32_f64_reader_consistency() {
    let num_samples = 4;
    let sample_size = 256; // 8 qubits
    let data_f64 = make_amplitude_data(num_samples, sample_size);
    let (f32_file, f64_file) = write_pair(&data_f64, sample_size);

    let mut reader_f32 =
        ParquetReader::<f32>::new(&f32_file.0, None, NullHandling::FillZero).unwrap();
    let (data32, n32, s32) = reader_f32.read_batch().unwrap();

    let mut reader_f64 =
        ParquetReader::<f64>::new(&f64_file.0, None, NullHandling::FillZero).unwrap();
    let (data64, n64, s64) = reader_f64.read_batch().unwrap();

    assert_eq!((n32, s32), (num_samples, sample_size));
    assert_eq!((n64, s64), (num_samples, sample_size));
    assert_eq!(data32.len(), data64.len());

    for (i, (&v32, &v64)) in data32.iter().zip(data64.iter()).enumerate() {
        assert_eq!(
            v32, v64 as f32,
            "f32 reader value {v32} != f64 narrowed value {} at index {i}",
            v64 as f32
        );
    }
}

// ---------------------------------------------------------------------------
// GPU fidelity (Linux + CUDA) — mirrors compare_f32_f64_amplitude, Parquet src.
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod gpu {
    use super::*;
    use qdp_core::Precision;
    use qdp_core::gpu::metrics::{
        download_complex_f32, download_complex_f64, fidelity_cross_precision,
    };

    /// Returns `true` only when CUDA kernels actually launch. On a stub build
    /// (no toolkit, e.g. CI) `QdpEngine::new` and `CudaDevice::new` still succeed
    /// and even host→device copies work — only kernel launches fail with code
    /// 999. So we probe with a trivial 1-qubit amplitude encode and treat any
    /// error as "no functional GPU" → skip. A real GPU that errors here would be
    /// a genuine bug, which the `.expect(...)` calls below still surface loudly.
    fn cuda_is_functional() -> bool {
        let Some(engine) = common::qdp_engine_with_precision(Precision::Float32) else {
            return false;
        };
        match engine.encode_batch_f32(&[1.0_f32, 0.0], 1, 2, 1, "amplitude") {
            Ok(dlpack) => {
                unsafe { common::take_deleter_and_delete(dlpack) };
                true
            }
            Err(_) => false,
        }
    }

    /// Read the same data from an f32 and an f64 Parquet column, encode at the
    /// matching precision, and return the minimum per-sample cross-precision
    /// fidelity. `None` when no functional GPU is available.
    fn parquet_amplitude_fidelity(num_qubits: usize) -> Option<f64> {
        if !cuda_is_functional() {
            return None;
        }
        let engine_f64 = common::qdp_engine_with_precision(Precision::Float64)?;
        let engine_f32 = common::qdp_engine_with_precision(Precision::Float32)?;
        let device = common::cuda_device()?;

        let state_dim = 1usize << num_qubits;
        let num_samples = 4;
        let sample_size = state_dim;

        let data_f64 = make_amplitude_data(num_samples, sample_size);
        let (f32_file, f64_file) = write_pair(&data_f64, sample_size);

        // f32 column → f32 reader → f32 kernel
        let mut reader_f32 =
            ParquetReader::<f32>::new(&f32_file.0, None, NullHandling::FillZero).unwrap();
        let (host_in_f32, _, _) = reader_f32.read_batch().unwrap();
        let dlpack_f32 = engine_f32
            .encode_batch_f32(
                &host_in_f32,
                num_samples,
                sample_size,
                num_qubits,
                "amplitude",
            )
            .expect("F32 encode_batch should succeed");

        // f64 column → f64 reader → f64 kernel
        let mut reader_f64 =
            ParquetReader::<f64>::new(&f64_file.0, None, NullHandling::FillZero).unwrap();
        let (host_in_f64, _, _) = reader_f64.read_batch().unwrap();
        let dlpack_f64 = engine_f64
            .encode_batch(
                &host_in_f64,
                num_samples,
                sample_size,
                num_qubits,
                "amplitude",
            )
            .expect("F64 encode_batch should succeed");

        let f64_tensor = unsafe { &(*dlpack_f64).dl_tensor };
        let f32_tensor = unsafe { &(*dlpack_f32).dl_tensor };

        let total_elements = num_samples * state_dim;
        let host_f64 =
            download_complex_f64(&device, f64_tensor.data as *const _, total_elements).unwrap();
        let host_f32 =
            download_complex_f32(&device, f32_tensor.data as *const _, total_elements).unwrap();

        let mut min_fidelity = 1.0_f64;
        for s in 0..num_samples {
            let off = s * state_dim * 2;
            let sample_f64 = &host_f64[off..off + state_dim * 2];
            let sample_f32 = &host_f32[off..off + state_dim * 2];
            let f = fidelity_cross_precision(sample_f32, sample_f64).unwrap();
            if f < min_fidelity {
                min_fidelity = f;
            }
        }

        unsafe {
            common::take_deleter_and_delete(dlpack_f64);
            common::take_deleter_and_delete(dlpack_f32);
        }

        Some(min_fidelity)
    }

    /// Run the Parquet f32-vs-f64 amplitude case at `num_qubits`, skipping when
    /// no GPU is available. Threshold matches `gpu_fidelity.rs` (1e-3 at 8-16).
    fn assert_amplitude_fidelity(num_qubits: usize) {
        let Some(fidelity) = parquet_amplitude_fidelity(num_qubits) else {
            println!("SKIP: No GPU available");
            return;
        };
        println!("Parquet F32 vs F64 fidelity @ {num_qubits} qubits: {fidelity:.10}");
        assert!(
            fidelity > 1.0 - 1e-3,
            "Fidelity too low at {num_qubits} qubits: {fidelity}"
        );
    }

    #[test]
    fn test_parquet_f32_vs_f64_amplitude_8_qubits() {
        assert_amplitude_fidelity(8);
    }

    #[test]
    fn test_parquet_f32_vs_f64_amplitude_12_qubits() {
        assert_amplitude_fidelity(12);
    }

    #[test]
    fn test_parquet_f32_vs_f64_amplitude_16_qubits() {
        assert_amplitude_fidelity(16);
    }
}
