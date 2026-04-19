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
