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

// Amplitude encoding: direct state injection with L2 normalization

use std::sync::Arc;
use cudarc::driver::CudaDevice;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;
use super::QuantumEncoder;

#[cfg(target_os = "linux")]
use std::ffi::c_void;
#[cfg(target_os = "linux")]
use cudarc::driver::DevicePtr;
#[cfg(target_os = "linux")]
use qdp_kernels::launch_amplitude_encode;

use crate::preprocessing::Preprocessor;

/// Amplitude encoding: data → normalized quantum amplitudes
///
/// Steps: L2 norm (CPU) → GPU allocation → CUDA kernel (normalize + pad)
/// Fast: ~50-100x vs circuit-based methods
pub struct AmplitudeEncoder;

impl QuantumEncoder for AmplitudeEncoder {
    fn encode(
        &self,
        _device: &Arc<CudaDevice>,
        host_data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        // Validate qubits (max 30 = 16GB GPU memory)
        Preprocessor::validate_input(host_data, num_qubits)?;
        let norm = Preprocessor::calculate_l2_norm(host_data)?;
        let state_len = 1 << num_qubits;

        #[cfg(target_os = "linux")]
        {
            // Allocate GPU state vector
            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new(_device, num_qubits)?
            };

            // Copy input data to GPU (synchronous, zero-copy from slice)
            // TODO : Use async CUDA streams for pipeline overlap
            let input_slice = {
                crate::profile_scope!("GPU::H2DCopy");
                _device.htod_sync_copy(host_data)
                    .map_err(|e| MahoutError::MemoryAllocation(format!("Failed to allocate input buffer: {:?}", e)))?
            };

            // Launch CUDA kernel (CPU-side launch only; execution is asynchronous)
            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    launch_amplitude_encode(
                        *input_slice.device_ptr() as *const f64,
                        state_vector.ptr() as *mut c_void,
                        host_data.len() as i32,
                        state_len as i32,
                        norm,
                        std::ptr::null_mut(), // default stream
                    )
                }
            };

            if ret != 0 {
                let error_msg = format!(
                    "Kernel launch failed with CUDA error code: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                );
                return Err(MahoutError::KernelLaunch(error_msg));
            }

            // Block until all work on the device is complete
            {
                crate::profile_scope!("GPU::Synchronize");
                _device
                    .synchronize()
                    .map_err(|e| MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e)))?;
            }

            Ok(state_vector)
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda("CUDA unavailable (non-Linux)".to_string()))
        }
    }

    fn name(&self) -> &'static str {
        "amplitude"
    }

    fn description(&self) -> &'static str {
        "Amplitude encoding with L2 normalization"
    }
}

/// Convert CUDA error code to human-readable string
#[cfg(target_os = "linux")]
fn cuda_error_to_string(code: i32) -> &'static str {
    match code {
        0 => "cudaSuccess",
        1 => "cudaErrorInvalidValue",
        2 => "cudaErrorMemoryAllocation",
        3 => "cudaErrorInitializationError",
        4 => "cudaErrorLaunchFailure",
        6 => "cudaErrorInvalidDevice",
        8 => "cudaErrorInvalidConfiguration",
        11 => "cudaErrorInvalidHostPointer",
        12 => "cudaErrorInvalidDevicePointer",
        17 => "cudaErrorInvalidMemcpyDirection",
        30 => "cudaErrorUnknown",
        _ => "Unknown CUDA error",
    }
}
