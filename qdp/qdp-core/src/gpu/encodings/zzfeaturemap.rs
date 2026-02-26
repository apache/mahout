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

//! ZZFeatureMap encoding: repeated H + Z + ZZ layers.
//!
//! This encoder implements a configurable ZZFeatureMap with repetition layers and
//! `full`, `linear`, or `circular` entanglement patterns.

use super::QuantumEncoder;
#[cfg(target_os = "linux")]
use crate::error::cuda_error_to_string;
use crate::error::{MahoutError, Result};
#[cfg(target_os = "linux")]
use crate::gpu::memory::Precision;
use crate::gpu::memory::GpuStateVector;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::gpu::memory::map_allocation_error;
#[cfg(target_os = "linux")]
use cudarc::driver::DevicePtr;
#[cfg(target_os = "linux")]
use std::ffi::c_void;

/// Entanglement pattern for ZZFeatureMap
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZZEntanglement {
    /// All pairs (i, j) where i < j: |E| = n*(n-1)/2
    Full,
    /// Nearest-neighbor pairs (i, i+1): |E| = n-1
    Linear,
    /// Linear + wrap-around (n-1, 0): |E| = n
    Circular,
}

impl ZZEntanglement {
    /// Parse entanglement pattern from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "full" => Some(Self::Full),
            "linear" => Some(Self::Linear),
            "circular" => Some(Self::Circular),
            _ => None,
        }
    }

    /// Number of entangled pairs for this pattern
    pub fn num_pairs(&self, num_qubits: usize) -> usize {
        match self {
            Self::Full => num_qubits * (num_qubits - 1) / 2,
            Self::Linear => {
                if num_qubits > 1 {
                    num_qubits - 1
                } else {
                    0
                }
            }
            Self::Circular => {
                if num_qubits > 1 {
                    num_qubits
                } else {
                    0
                }
            }
        }
    }

    /// Convert to integer mode for CUDA kernel
    pub fn to_mode(&self) -> i32 {
        match self {
            Self::Full => 0,
            Self::Linear => 1,
            Self::Circular => 2,
        }
    }
}

/// ZZFeatureMap encoder with configurable layers and entanglement.
pub struct ZZFeatureMapEncoder {
    /// Number of repetition layers
    reps: usize,
    /// Entanglement pattern
    entanglement: ZZEntanglement,
}

impl ZZFeatureMapEncoder {
    /// Create a ZZFeatureMap encoder with specified reps and entanglement.
    #[must_use]
    pub fn new(reps: usize, entanglement: ZZEntanglement) -> Self {
        Self { reps, entanglement }
    }

    /// Create a ZZFeatureMap encoder with full entanglement and reps=2 (Qiskit default).
    #[must_use]
    pub fn default_full() -> Self {
        Self {
            reps: 2,
            entanglement: ZZEntanglement::Full,
        }
    }

    /// Create a ZZFeatureMap encoder with linear entanglement.
    #[must_use]
    pub fn linear(reps: usize) -> Self {
        Self {
            reps,
            entanglement: ZZEntanglement::Linear,
        }
    }

    /// Create a ZZFeatureMap encoder with circular entanglement.
    #[must_use]
    pub fn circular(reps: usize) -> Self {
        Self {
            reps,
            entanglement: ZZEntanglement::Circular,
        }
    }

    /// Calculate the expected data length for this encoder configuration.
    fn expected_data_len(&self, num_qubits: usize) -> usize {
        let num_pairs = self.entanglement.num_pairs(num_qubits);
        self.reps * (num_qubits + num_pairs)
    }

    /// Get the number of parameters per layer.
    fn params_per_layer(&self, num_qubits: usize) -> usize {
        num_qubits + self.entanglement.num_pairs(num_qubits)
    }
}

impl QuantumEncoder for ZZFeatureMapEncoder {
    fn encode(
        &self,
        #[cfg(target_os = "linux")] device: &Arc<CudaDevice>,
        #[cfg(not(target_os = "linux"))] _device: &Arc<CudaDevice>,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        self.validate_input(data, num_qubits)?;
        let state_len = 1 << num_qubits;

        #[cfg(target_os = "linux")]
        {
            let input_bytes = std::mem::size_of_val(data);
            let data_gpu = {
                crate::profile_scope!("GPU::H2D_ZZFeatureMapData");
                device.htod_sync_copy(data).map_err(|e| {
                    map_allocation_error(
                        input_bytes,
                        "ZZFeatureMap input upload",
                        Some(num_qubits),
                        e,
                    )
                })?
            };

            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new(device, num_qubits, Precision::Float64)?
            };

            let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
                MahoutError::InvalidInput(
                    "State vector precision mismatch (expected float64 buffer)".to_string(),
                )
            })?;

            let params_per_layer = self.params_per_layer(num_qubits) as u32;

            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    qdp_kernels::launch_zzfeaturemap_encode(
                        *data_gpu.device_ptr() as *const f64,
                        state_ptr as *mut c_void,
                        state_len,
                        num_qubits as u32,
                        self.reps as u32,
                        params_per_layer,
                        self.entanglement.to_mode(),
                        std::ptr::null_mut(),
                    )
                }
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "ZZFeatureMap encoding kernel failed with CUDA error code: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }

            {
                crate::profile_scope!("GPU::Synchronize");
                device.synchronize().map_err(|e| {
                    MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e))
                })?;
            }

            Ok(state_vector)
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda(
                "CUDA unavailable (non-Linux stub)".to_string(),
            ))
        }
    }

    fn validate_input(&self, data: &[f64], num_qubits: usize) -> Result<()> {
        if num_qubits == 0 {
            return Err(MahoutError::InvalidInput(
                "Number of qubits must be at least 1".to_string(),
            ));
        }
        if num_qubits > 30 {
            return Err(MahoutError::InvalidInput(format!(
                "Number of qubits {} exceeds practical limit of 30",
                num_qubits
            )));
        }
        if self.reps == 0 {
            return Err(MahoutError::InvalidInput(
                "Number of repetition layers (reps) must be at least 1".to_string(),
            ));
        }

        let expected_len = self.expected_data_len(num_qubits);
        if data.len() != expected_len {
            return Err(MahoutError::InvalidInput(format!(
                "ZZFeatureMap ({:?}, reps={}) expects {} values for {} qubits, got {}",
                self.entanglement,
                self.reps,
                expected_len,
                num_qubits,
                data.len()
            )));
        }

        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(MahoutError::InvalidInput(format!(
                    "Parameter at index {} must be finite, got {}",
                    i, val
                )));
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "zzfeaturemap"
    }

    fn description(&self) -> &'static str {
        "ZZFeatureMap encoding: quantum feature map with repeated H + Z + ZZ layers"
    }
}

/// Parse ZZFeatureMap configuration from encoding method string.
///
/// Supported formats:
/// - `"zzfeaturemap"` → full entanglement, reps=2
/// - `"zzfeaturemap-full"` → full entanglement, reps=2
/// - `"zzfeaturemap-linear"` → linear entanglement, reps=2
/// - `"zzfeaturemap-circular"` → circular entanglement, reps=2
/// - `"zzfeaturemap-full-reps3"` → full entanglement, reps=3
/// - `"zzfeaturemap-linear-reps1"` → linear entanglement, reps=1
pub fn parse_zzfeaturemap_config(name: &str) -> Option<ZZFeatureMapEncoder> {
    let name_lower = name.to_lowercase();

    if !name_lower.starts_with("zzfeaturemap") && !name_lower.starts_with("zz_feature_map") {
        return None;
    }

    // Default values
    let mut entanglement = ZZEntanglement::Full;
    let mut reps: usize = 2;

    // Parse parts after the prefix
    let parts: Vec<&str> = name_lower
        .trim_start_matches("zzfeaturemap")
        .trim_start_matches("zz_feature_map")
        .split('-')
        .filter(|s| !s.is_empty())
        .collect();

    for part in parts {
        if let Some(ent) = ZZEntanglement::from_str(part) {
            entanglement = ent;
        } else if let Some(r) = part.strip_prefix("reps") {
            if let Ok(r_val) = r.parse::<usize>() {
                reps = r_val;
            }
        }
    }

    Some(ZZFeatureMapEncoder::new(reps, entanglement))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entanglement_num_pairs() {
        assert_eq!(ZZEntanglement::Full.num_pairs(2), 1);
        assert_eq!(ZZEntanglement::Full.num_pairs(3), 3);
        assert_eq!(ZZEntanglement::Full.num_pairs(4), 6);

        assert_eq!(ZZEntanglement::Linear.num_pairs(2), 1);
        assert_eq!(ZZEntanglement::Linear.num_pairs(3), 2);
        assert_eq!(ZZEntanglement::Linear.num_pairs(4), 3);

        assert_eq!(ZZEntanglement::Circular.num_pairs(2), 2);
        assert_eq!(ZZEntanglement::Circular.num_pairs(3), 3);
        assert_eq!(ZZEntanglement::Circular.num_pairs(4), 4);
    }

    #[test]
    fn test_expected_data_len() {
        // 2 qubits, full, reps=2: 2 × (2 + 1) = 6
        let enc = ZZFeatureMapEncoder::new(2, ZZEntanglement::Full);
        assert_eq!(enc.expected_data_len(2), 6);

        // 3 qubits, full, reps=1: 1 × (3 + 3) = 6
        let enc = ZZFeatureMapEncoder::new(1, ZZEntanglement::Full);
        assert_eq!(enc.expected_data_len(3), 6);

        // 3 qubits, linear, reps=2: 2 × (3 + 2) = 10
        let enc = ZZFeatureMapEncoder::new(2, ZZEntanglement::Linear);
        assert_eq!(enc.expected_data_len(3), 10);
    }

    #[test]
    fn test_parse_config() {
        let enc = parse_zzfeaturemap_config("zzfeaturemap").unwrap();
        assert_eq!(enc.reps, 2);
        assert_eq!(enc.entanglement, ZZEntanglement::Full);

        let enc = parse_zzfeaturemap_config("zzfeaturemap-linear").unwrap();
        assert_eq!(enc.entanglement, ZZEntanglement::Linear);

        let enc = parse_zzfeaturemap_config("zzfeaturemap-full-reps3").unwrap();
        assert_eq!(enc.reps, 3);
        assert_eq!(enc.entanglement, ZZEntanglement::Full);

        let enc = parse_zzfeaturemap_config("zzfeaturemap-circular-reps1").unwrap();
        assert_eq!(enc.reps, 1);
        assert_eq!(enc.entanglement, ZZEntanglement::Circular);
    }

    #[test]
    fn test_validate_input_errors() {
        let enc = ZZFeatureMapEncoder::default_full();

        // Wrong length
        let result = enc.validate_input(&[0.0; 5], 2);
        assert!(result.is_err());

        // NaN value
        let result = enc.validate_input(&[f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0], 2);
        assert!(result.is_err());

        // Zero qubits
        let result = enc.validate_input(&[], 0);
        assert!(result.is_err());
    }
}
