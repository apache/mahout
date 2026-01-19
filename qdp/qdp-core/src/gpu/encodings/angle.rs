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

// Angle encoding: map per-qubit angles to product state amplitudes.

use super::QuantumEncoder;
#[cfg(target_os = "linux")]
use crate::error::cuda_error_to_string;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::gpu::memory::map_allocation_error;
#[cfg(target_os = "linux")]
use cudarc::driver::DevicePtr;
#[cfg(target_os = "linux")]
use std::ffi::c_void;

/// Angle encoding: each qubit uses one rotation angle to form a product state.
pub struct AngleEncoder;

impl QuantumEncoder for AngleEncoder {
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
            let angles_gpu = {
                crate::profile_scope!("GPU::H2D_Angles");
                device.htod_sync_copy(data).map_err(|e| {
                    map_allocation_error(input_bytes, "angle input upload", Some(num_qubits), e)
                })?
            };

            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new(device, num_qubits)?
            };

            let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
                MahoutError::InvalidInput(
                    "State vector precision mismatch (expected float64 buffer)".to_string(),
                )
            })?;

            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    qdp_kernels::launch_angle_encode(
                        *angles_gpu.device_ptr() as *const f64,
                        state_ptr as *mut c_void,
                        state_len,
                        num_qubits as u32,
                        std::ptr::null_mut(),
                    )
                }
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Angle encoding kernel failed with CUDA error code: {} ({})",
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

    /// Encode multiple angle samples in a single GPU allocation and kernel launch
    #[cfg(target_os = "linux")]
    fn encode_batch(
        &self,
        device: &Arc<CudaDevice>,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        crate::profile_scope!("AngleEncoder::encode_batch");

        if sample_size != num_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "Angle encoding expects sample_size={} (one angle per qubit), got {}",
                num_qubits, sample_size
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

        if num_qubits == 0 || num_qubits > 30 {
            return Err(MahoutError::InvalidInput(format!(
                "Number of qubits {} must be between 1 and 30",
                num_qubits
            )));
        }

        for (i, &val) in batch_data.iter().enumerate() {
            if !val.is_finite() {
                let sample_idx = i / sample_size;
                let angle_idx = i % sample_size;
                return Err(MahoutError::InvalidInput(format!(
                    "Sample {} angle {} must be finite, got {}",
                    sample_idx, angle_idx, val
                )));
            }
        }

        let state_len = 1 << num_qubits;

        let batch_state_vector = {
            crate::profile_scope!("GPU::AllocBatch");
            GpuStateVector::new_batch(device, num_samples, num_qubits)?
        };

        let input_bytes = std::mem::size_of_val(batch_data);
        let angles_gpu = {
            crate::profile_scope!("GPU::H2D_BatchAngles");
            device.htod_sync_copy(batch_data).map_err(|e| {
                map_allocation_error(input_bytes, "angle batch upload", Some(num_qubits), e)
            })?
        };

        let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "Batch state vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        {
            crate::profile_scope!("GPU::BatchKernelLaunch");
            let ret = unsafe {
                qdp_kernels::launch_angle_encode_batch(
                    *angles_gpu.device_ptr() as *const f64,
                    state_ptr as *mut c_void,
                    num_samples,
                    state_len,
                    num_qubits as u32,
                    std::ptr::null_mut(),
                )
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Batch angle encoding kernel failed: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }
        }

        {
            crate::profile_scope!("GPU::Synchronize");
            device
                .synchronize()
                .map_err(|e| MahoutError::Cuda(format!("Sync failed: {:?}", e)))?;
        }

        Ok(batch_state_vector)
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
        if data.len() != num_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "Angle encoding expects {} values (one per qubit), got {}",
                num_qubits,
                data.len()
            )));
        }
        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(MahoutError::InvalidInput(format!(
                    "Angle at index {} must be finite, got {}",
                    i, val
                )));
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "angle"
    }

    fn description(&self) -> &'static str {
        "Angle encoding: per-qubit rotations into a product state"
    }
}
