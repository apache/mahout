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
use crate::gpu::pipeline::run_dual_stream_pipeline;
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

            // SSS-Tier Optimization: Async Pipeline for large data
            // For small data (< 1MB), use synchronous path to avoid stream overhead
            // For large data, use dual-stream async pipeline for maximum throughput
            const ASYNC_THRESHOLD: usize = 1024 * 1024 / std::mem::size_of::<f64>(); // 1MB threshold

            if host_data.len() < ASYNC_THRESHOLD {
                // Synchronous path for small data (avoids stream overhead)
            let input_slice = {
                crate::profile_scope!("GPU::H2DCopy");
                _device.htod_sync_copy(host_data)
                    .map_err(|e| MahoutError::MemoryAllocation(format!("Failed to allocate input buffer: {:?}", e)))?
            };

            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    launch_amplitude_encode(
                        *input_slice.device_ptr() as *const f64,
                        state_vector.ptr() as *mut c_void,
                            host_data.len(),
                            state_len,
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

            {
                crate::profile_scope!("GPU::Synchronize");
                _device
                    .synchronize()
                    .map_err(|e| MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e)))?;
                }
            } else {
                // Async Pipeline path for large data
                Self::encode_async_pipeline(_device, host_data, num_qubits, state_len, norm, &state_vector)?;
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

impl AmplitudeEncoder {
    /// Async pipeline encoding for large data (SSS-tier optimization)
    ///
    /// Uses the generic dual-stream pipeline infrastructure to overlap
    /// data transfer and computation. The pipeline handles all the
    /// streaming mechanics, while this method focuses on the amplitude
    /// encoding kernel logic.
    #[cfg(target_os = "linux")]
    fn encode_async_pipeline(
        device: &Arc<CudaDevice>,
        host_data: &[f64],
        _num_qubits: usize,
        state_len: usize,
        norm: f64,
        state_vector: &GpuStateVector,
    ) -> Result<()> {
        // Use generic pipeline infrastructure
        // The closure handles amplitude-specific kernel launch logic
        run_dual_stream_pipeline(device, host_data, |stream, input_ptr, chunk_offset, chunk_len| {
            // Calculate offset pointer for state vector (type-safe pointer arithmetic)
            // Offset is in complex numbers (CuDoubleComplex), not f64 elements
            let state_ptr_offset = unsafe {
                state_vector.ptr().cast::<u8>()
                    .add(chunk_offset * std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
                    .cast::<std::ffi::c_void>()
            };

            // Launch amplitude encoding kernel on the provided stream
            let ret = unsafe {
                launch_amplitude_encode(
                    input_ptr,
                    state_ptr_offset,
                    chunk_len,
                    state_len,
                    norm,
                    stream.stream as *mut c_void,
                )
            };

            if ret != 0 {
                let error_msg = format!(
                    "Kernel launch failed with CUDA error code: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                );
                return Err(MahoutError::KernelLaunch(error_msg));
            }

            Ok(())
        })?;

        // CRITICAL FIX: Handle padding for uninitialized memory
        // Since we use alloc() (uninitialized), we must zero-fill any tail region
        // that wasn't written by the pipeline. This ensures correctness when
        // host_data.len() < state_len (e.g., 1000 elements in a 1024-element state).
        let data_len = host_data.len();
        if data_len < state_len {
            let padding_start = data_len;
            let padding_elements = state_len - padding_start;
            let padding_bytes = padding_elements * std::mem::size_of::<qdp_kernels::CuDoubleComplex>();

            // Calculate tail pointer (in complex numbers)
            let tail_ptr = unsafe {
                state_vector.ptr().add(padding_start) as *mut c_void
            };

            // Zero-fill padding region using CUDA Runtime API
            // Use default stream since pipeline streams are already synchronized
            unsafe {
                unsafe extern "C" {
                    fn cudaMemsetAsync(
                        devPtr: *mut c_void,
                        value: i32,
                        count: usize,
                        stream: *mut c_void,
                    ) -> i32;
                }

                let result = cudaMemsetAsync(
                    tail_ptr,
                    0,
                    padding_bytes,
                    std::ptr::null_mut(), // default stream
                );

                if result != 0 {
                    return Err(MahoutError::Cuda(
                        format!("Failed to zero-fill padding region: {} ({})",
                                result, cuda_error_to_string(result))
                    ));
                }
            }

            // Synchronize to ensure padding is complete before returning
            device.synchronize()
                .map_err(|e| MahoutError::Cuda(format!("Failed to sync after padding: {:?}", e)))?;
        }

        Ok(())
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
