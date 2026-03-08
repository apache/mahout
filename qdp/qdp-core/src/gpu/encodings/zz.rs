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

//! ZZFeatureMap encoding implementation.

use super::QuantumEncoder;
#[cfg(target_os = "linux")]
use crate::error::cuda_error_to_string;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::{GpuStateVector, Precision};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::gpu::memory::map_allocation_error;
#[cfg(target_os = "linux")]
use cudarc::driver::DevicePtr;
#[cfg(target_os = "linux")]
use std::ffi::c_void;

/// ZZFeatureMap encoding: maps classical features to quantum states using H, RZ, and RZZ gates.
/// Supports multiple repetition layers.
pub struct ZzEncoder {
    num_layers: usize,
}

impl ZzEncoder {
    /// Create a new ZZFeatureMap encoder with the specified number of layers.
    pub fn new(num_layers: usize) -> Self {
        Self { num_layers }
    }
}

impl Default for ZzEncoder {
    fn default() -> Self {
        Self::new(2) // Default in many QML libraries
    }
}

impl QuantumEncoder for ZzEncoder {
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
                crate::profile_scope!("GPU::H2D_ZzData");
                device.htod_sync_copy(data).map_err(|e| {
                    map_allocation_error(input_bytes, "ZZ input upload", Some(num_qubits), e)
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

            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    qdp_kernels::launch_zz_encode(
                        *data_gpu.device_ptr() as *const f64,
                        state_ptr as *mut c_void,
                        state_len,
                        num_qubits as u32,
                        self.num_layers as u32,
                        std::ptr::null_mut(),
                    )
                }
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "ZZ encoding kernel failed with CUDA error code: {} ({})",
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

    #[cfg(target_os = "linux")]
    fn encode_batch(
        &self,
        device: &Arc<CudaDevice>,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        crate::profile_scope!("ZzEncoder::encode_batch");

        if sample_size != num_qubits {
             return Err(MahoutError::InvalidInput(format!(
                "ZZFeatureMap encoding expects sample_size={} for {} qubits, got {}",
                num_qubits,
                num_qubits,
                sample_size
            )));
        }

        if batch_data.len() != num_samples * sample_size {
            return Err(MahoutError::InvalidInput(format!(
                "Batch data length {} doesn't match num_samples {} * sample_size {}",
                batch_data.len(),
                num_samples,
                sample_size
            )));
        }

        let state_len = 1 << num_qubits;
        let batch_state_vector = GpuStateVector::new_batch(device, num_samples, num_qubits, Precision::Float64)?;

        let input_bytes = std::mem::size_of_val(batch_data);
        let data_gpu = {
            crate::profile_scope!("GPU::H2D_BatchZzData");
            device.htod_sync_copy(batch_data).map_err(|e| {
                map_allocation_error(input_bytes, "ZZ batch upload", Some(num_qubits), e)
            })?
        };

        let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "Batch state vector precision mismatch".to_string(),
            )
        })?;

        {
            crate::profile_scope!("GPU::BatchKernelLaunch");
            let ret = unsafe {
                qdp_kernels::launch_zz_encode_batch(
                    *data_gpu.device_ptr() as *const f64,
                    state_ptr as *mut c_void,
                    num_samples,
                    state_len,
                    num_qubits as u32,
                    sample_size as u32,
                    self.num_layers as u32,
                    std::ptr::null_mut(),
                )
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Batch ZZ encoding kernel failed: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }
        }

        device.synchronize().map_err(|e| MahoutError::Cuda(format!("Sync failed: {:?}", e)))?;
        Ok(batch_state_vector)
    }

    fn validate_input(&self, data: &[f64], num_qubits: usize) -> Result<()> {
        if num_qubits == 0 {
            return Err(MahoutError::InvalidInput("Number of qubits must be at least 1".to_string()));
        }
        if data.len() != num_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "ZZFeatureMap expects {} values (one per qubit), got {}",
                num_qubits, data.len()
            )));
        }
        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(MahoutError::InvalidInput(format!("Parameter at index {} is not finite: {}", i, val)));
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "zz"
    }

    fn description(&self) -> &'static str {
        "ZZFeatureMap: Second-order expansion encoding with RZ and ZZ interactions"
    }
}
