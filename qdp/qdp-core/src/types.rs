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

//! Canonical domain types for encodings and element dtypes (`Dtype`).
//!
//! ## `Encoding::supports_f32`
//!
//! A future shape of this API may return true for amplitude, angle, and basis once each encoder
//! has a batch float32 GPU path. **Today only amplitude implements**
//! [`QuantumEncoder::encode_batch_f32`] for the synthetic prefetch pipeline, so
//! [`Encoding::supports_f32`](Encoding::supports_f32) stays amplitude-only and
//! [`crate::pipeline_runner::PipelineConfig::normalize`] avoids routing other encodings through
//! `encode_batch_f32`. Widen this method when angle/basis gain real `encode_batch_f32`
//! implementations.

use crate::error::{MahoutError, Result};
use crate::gpu::encodings::{
    AmplitudeEncoder, AngleEncoder, BasisEncoder, PhaseEncoder, QuantumEncoder, iqp_full_encoder,
    iqp_z_encoder,
};

/// Dtype for pipeline configuration (re-export of [`crate::gpu::memory::Precision`]).
pub use crate::gpu::memory::Precision as Dtype;

impl crate::gpu::memory::Precision {
    /// Parse dtype from a short user string (case-insensitive, trimmed).
    pub fn from_str_ci(s: &str) -> Result<Self> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("f32")
            || t.eq_ignore_ascii_case("float32")
            || t.eq_ignore_ascii_case("float")
        {
            Ok(Self::Float32)
        } else if t.eq_ignore_ascii_case("f64")
            || t.eq_ignore_ascii_case("float64")
            || t.eq_ignore_ascii_case("double")
        {
            Ok(Self::Float64)
        } else {
            Err(MahoutError::InvalidInput(format!(
                "Unknown dtype: {s}. Use 'f32' or 'f64'."
            )))
        }
    }

    /// Element size in bytes for real scalar components (f32/f64).
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float64 => 8,
        }
    }
}

/// Quantum encoding method (canonical; parse user strings once at API boundaries).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    Amplitude,
    Angle,
    Basis,
    Iqp,
    IqpZ,
    Phase,
}

impl Encoding {
    /// Parse encoding name (case-insensitive ASCII, stack buffer; no heap allocation).
    pub fn from_str_ci(s: &str) -> Result<Self> {
        let mut buf = [0u8; 16];
        let bytes = s.as_bytes();
        if bytes.len() > buf.len() {
            return Err(MahoutError::InvalidInput(format!(
                "Unknown encoding: {s}. Available: amplitude, angle, basis, iqp, iqp-z, phase"
            )));
        }
        for (i, b) in bytes.iter().enumerate() {
            buf[i] = b.to_ascii_lowercase();
        }
        match &buf[..bytes.len()] {
            b"amplitude" => Ok(Self::Amplitude),
            b"angle" => Ok(Self::Angle),
            b"basis" => Ok(Self::Basis),
            b"iqp" => Ok(Self::Iqp),
            b"iqp-z" => Ok(Self::IqpZ),
            b"phase" => Ok(Self::Phase),
            _ => Err(MahoutError::InvalidInput(format!(
                "Unknown encoding: {s}. Available: amplitude, angle, basis, iqp, iqp-z, phase"
            ))),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Amplitude => "amplitude",
            Self::Angle => "angle",
            Self::Basis => "basis",
            Self::Iqp => "iqp",
            Self::IqpZ => "iqp-z",
            Self::Phase => "phase",
        }
    }

    /// Input feature dimension per sample for this encoding and qubit count.
    ///
    /// Matches each encoder's `expected_data_len` / `sample_size` contract:
    /// - `Amplitude`: full state vector (`2^n`)
    /// - `Angle` / `IqpZ` / `Phase`: one value per qubit (`n`)
    /// - `Iqp`: single-qubit + pairwise ZZ terms (`n + n*(n-1)/2`)
    /// - `Basis`: single integer index (`1`)
    #[must_use]
    pub const fn vector_len(self, num_qubits: u32) -> usize {
        let n = num_qubits as usize;
        match self {
            Self::Amplitude => 1 << n,
            Self::Angle | Self::IqpZ | Self::Phase => n,
            Self::Iqp => n + n * n.saturating_sub(1) / 2,
            Self::Basis => 1,
        }
    }

    /// Whether the **synthetic batch pipeline** may keep [`crate::gpu::memory::Precision::Float32`]
    /// end-to-end (prefetched host `Vec<f32>` plus [`crate::QdpEngine::encode_batch_f32`]).
    ///
    /// Returns true for encodings whose batch host fill and `encode_batch_f32` paths are wired
    /// end-to-end: amplitude, angle, basis. IQP / IQP-Z / Phase still normalize to `Float64`
    /// in [`crate::pipeline_runner::PipelineConfig::normalize`] until their batch f32 GPU
    /// paths exist.
    #[must_use]
    pub const fn supports_f32(self) -> bool {
        matches!(self, Self::Amplitude | Self::Angle | Self::Basis)
    }

    /// Static encoder dispatch (no per-call heap allocation).
    #[must_use]
    pub fn encoder(self) -> &'static dyn QuantumEncoder {
        match self {
            Self::Amplitude => &AmplitudeEncoder,
            Self::Angle => &AngleEncoder,
            Self::Basis => &BasisEncoder,
            Self::Iqp => iqp_full_encoder(),
            Self::IqpZ => iqp_z_encoder(),
            Self::Phase => &PhaseEncoder,
        }
    }
}

/// Compute backend used for the Rust-side encoding pipeline.
///
/// Today this is a thin, 1-D classification used by callers (Python bindings,
/// benchmarks, future planning layers) to ask "what compute target is the
/// Rust pipeline running against?". It is intentionally **not** plumbed through
/// [`crate::pipeline_runner::PipelineConfig`] as a dispatch axis: there is
/// currently only one Rust-side encoder implementation (CUDA / nvcc), so a 2-D
/// `(Encoding, Backend) → &'static dyn QuantumEncoder` table would have nothing
/// extra to dispatch to.
///
/// The AMD / Triton path that exists upstream (`qumat_qdp/backends/amd.py`,
/// added in #1158) lives entirely on the Python side. It does not have a
/// matching Rust-level [`crate::QuantumEncoder`] implementation, so it is
/// **not** modelled here as a `Backend` variant. When (and if) a Rust-AMD
/// path is proposed, add the variant and grow the dispatch table at the same
/// time — not before.
///
/// `TorchRef` represents the pure-PyTorch reference implementations
/// (`torch_ref.py`, added in #1189) used by CI machines without a CUDA GPU.
/// It is included so callers can distinguish "running on CUDA" from "running
/// the pure-PyTorch fallback" without sniffing the engine state.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// NVIDIA CUDA via the in-tree CUDA kernels (the historical default).
    CudaNvidia,
    /// Pure-PyTorch reference implementation for CPU / non-CUDA CI lanes.
    TorchRef,
}

impl Backend {
    /// Stable lower-case name suitable for logging and user-facing error messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CudaNvidia => "cuda-nvidia",
            Self::TorchRef => "torch-ref",
        }
    }

    /// Parse a backend name (case-insensitive, ASCII).
    ///
    /// Accepts the canonical names returned by [`as_str`](Self::as_str) plus a few
    /// convenience aliases (`cuda`, `nvidia`, `torch`, `cpu`). Unknown names that
    /// resemble unsupported backends (notably `amd` / `triton`) get a dedicated
    /// error message pointing to the Python-side AMD path so users don't think
    /// the Rust core silently ignores them.
    pub fn from_str_ci(s: &str) -> Result<Self> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("cuda-nvidia")
            || t.eq_ignore_ascii_case("cuda")
            || t.eq_ignore_ascii_case("nvidia")
        {
            Ok(Self::CudaNvidia)
        } else if t.eq_ignore_ascii_case("torch-ref")
            || t.eq_ignore_ascii_case("torch")
            || t.eq_ignore_ascii_case("torchref")
            || t.eq_ignore_ascii_case("cpu")
        {
            Ok(Self::TorchRef)
        } else if t.eq_ignore_ascii_case("amd")
            || t.eq_ignore_ascii_case("triton")
            || t.eq_ignore_ascii_case("triton-amd")
        {
            Err(MahoutError::InvalidInput(
                "AMD / Triton backend is not selectable from the Rust core; use the \
                 Python-side `qumat_qdp.backends.amd` path instead. The Rust `Backend` \
                 enum will gain an `AMD` variant once a Rust-side AMD encoder lands."
                    .to_string(),
            ))
        } else {
            Err(MahoutError::InvalidInput(format!(
                "Unknown backend: {s}. Available: cuda-nvidia (alias: cuda), torch-ref (alias: torch, cpu)."
            )))
        }
    }

    /// Best-guess backend for the current build / host.
    ///
    /// Returns [`Backend::CudaNvidia`] on Linux (the platform that compiles the
    /// CUDA kernels) and [`Backend::TorchRef`] elsewhere. This does **not** probe
    /// for a working CUDA runtime — it is a compile-time classification, not a
    /// runtime capability check. Use [`QdpEngine::new`](crate::QdpEngine::new) to
    /// fail with a clear error when a CUDA device is unavailable at runtime.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self::CudaNvidia
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self::TorchRef
        }
    }
}
