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
use arrow::array::{Array, Float64Array};
use cudarc::driver::CudaDevice;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;
use crate::gpu::pipeline::run_dual_stream_pipeline;
use crate::gpu::pool::StagingBufferPool;
use super::QuantumEncoder;

#[cfg(target_os = "linux")]
use std::ffi::c_void;
#[cfg(target_os = "linux")]
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
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
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
                GpuStateVector::new(device, num_qubits)?
            };

            // Async Pipeline for large data
            // For small data (< 1MB), use synchronous path to avoid stream overhead
            // For large data, use dual-stream async pipeline for maximum throughput
            const ASYNC_THRESHOLD: usize = 1024 * 1024 / std::mem::size_of::<f64>(); // 1MB threshold

            if host_data.len() < ASYNC_THRESHOLD {
                // Synchronous path for small data (avoids stream overhead)
                let input_bytes = host_data.len() * std::mem::size_of::<f64>();

                // Acquire buffer (RAII: auto-released on drop)
                let mut _staging_buffer = {
                    crate::profile_scope!("Pool::Acquire");
                    pool.acquire(input_bytes)?
                };

                // Convert f64 to u8 view (zero-copy) and copy to device
                let src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(host_data.as_ptr() as *const u8, input_bytes)
                };

                {
                    crate::profile_scope!("GPU::H2DCopy");
                    device.htod_sync_copy_into(src_bytes, &mut _staging_buffer.slice_mut(0..input_bytes))
                        .map_err(|e| MahoutError::Cuda(format!("H2D copy failed: {:?}", e)))?;
                }

                let ret = {
                    crate::profile_scope!("GPU::KernelLaunch");
                    unsafe {
                        launch_amplitude_encode(
                            _staging_buffer.device_ptr_u8() as *const f64,
                            state_vector.ptr() as *mut c_void,
                            host_data.len(),
                            state_len,
                            norm,
                            std::ptr::null_mut(), // default stream
                        )
                    }
                };

                // BufferGuard auto-releases buffer on drop

                if ret != 0 {
                    let error_msg = if ret == 2 {
                        format!(
                            "Kernel launch reported cudaErrorMemoryAllocation (likely OOM) while encoding {} elements into 2^{} state.",
                            host_data.len(),
                            num_qubits,
                        )
                    } else {
                        format!(
                            "Kernel launch failed with CUDA error code: {} ({})",
                            ret,
                            cuda_error_to_string(ret)
                        )
                    };
                    return Err(MahoutError::KernelLaunch(error_msg));
                }

                {
                    crate::profile_scope!("GPU::Synchronize");
                    device
                        .synchronize()
                        .map_err(|e| MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e)))?;
                }
            } else {
                // Async Pipeline path for large data
                Self::encode_async_pipeline(device, host_data, num_qubits, state_len, norm, &state_vector)?;
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

    /// Override to avoid intermediate Vec allocation. Processes chunks directly to GPU offsets.
    fn encode_chunked(
        &self,
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
        chunks: &[Float64Array],
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        #[cfg(target_os = "linux")]
        {
            let total_len: usize = chunks.iter().map(|c| c.len()).sum();
            let state_len = 1 << num_qubits;

            if total_len == 0 {
                return Err(MahoutError::InvalidInput("Input chunks cannot be empty".to_string()));
            }
            if total_len > state_len {
                return Err(MahoutError::InvalidInput(
                    format!("Total data length {} exceeds state vector size {}", total_len, state_len)
                ));
            }
            if num_qubits == 0 || num_qubits > 30 {
                return Err(MahoutError::InvalidInput(
                    format!("Number of qubits {} must be between 1 and 30", num_qubits)
                ));
            }

            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new(device, num_qubits)?
            };

            // Require pre-processed data (no nulls)
            for chunk in chunks {
                if chunk.null_count() > 0 {
                    return Err(MahoutError::InvalidInput(
                        format!("Chunk contains {} null values. Data must be pre-processed before encoding", chunk.null_count())
                    ));
                }
            }

            let norm = {
                crate::profile_scope!("CPU::L2Norm");
                let mut norm_sq = 0.0;
                for chunk in chunks {
                    norm_sq += chunk.values().iter().map(|&x| x * x).sum::<f64>();
                }
                let norm = norm_sq.sqrt();
                if norm == 0.0 {
                    return Err(MahoutError::InvalidInput("Input data has zero norm".to_string()));
                }
                norm
            };

            let mut current_offset = 0;
            for chunk in chunks {
                let chunk_len = chunk.len();
                if chunk_len == 0 {
                    continue;
                }

                let state_ptr_offset = unsafe {
                    state_vector.ptr().cast::<u8>()
                        .add(current_offset * std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
                        .cast::<std::ffi::c_void>()
                };

                let chunk_bytes = chunk_len * std::mem::size_of::<f64>();

                // Acquire buffer (RAII: auto-released on drop)
                let mut _chunk_buffer = {
                    crate::profile_scope!("Pool::Acquire");
                    pool.acquire(chunk_bytes)?
                };

                // Convert f64 to u8 view (zero-copy) and copy to device
                let src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        chunk.values().as_ptr() as *const u8,
                        chunk_bytes
                    )
                };

                {
                    crate::profile_scope!("GPU::ChunkH2DCopy");
                    device.htod_sync_copy_into(src_bytes, &mut _chunk_buffer.slice_mut(0..chunk_bytes))
                        .map_err(|e| MahoutError::MemoryAllocation(
                            format!("Failed to copy chunk: {:?}", e)
                        ))?;
                }

                {
                    crate::profile_scope!("GPU::KernelLaunch");
                    let ret = unsafe {
                        launch_amplitude_encode(
                            _chunk_buffer.device_ptr_u8() as *const f64,
                            state_ptr_offset,
                            chunk_len,
                            state_len,
                            norm,
                            std::ptr::null_mut(),
                        )
                    };

                    // BufferGuard auto-releases buffer on drop

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(
                            format!("Kernel launch failed: {} ({})", ret, cuda_error_to_string(ret))
                        ));
                    }
                }

                current_offset += chunk_len;
            }

            if total_len < state_len {
                let padding_bytes = (state_len - total_len) * std::mem::size_of::<qdp_kernels::CuDoubleComplex>();
                let tail_ptr = unsafe { state_vector.ptr().add(total_len) as *mut c_void };

                unsafe {
                    unsafe extern "C" {
                        fn cudaMemsetAsync(devPtr: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;
                    }
                    let result = cudaMemsetAsync(tail_ptr, 0, padding_bytes, std::ptr::null_mut());
                    if result != 0 {
                        return Err(MahoutError::Cuda(
                            format!("Failed to zero-fill padding: {} ({})", result, cuda_error_to_string(result))
                        ));
                    }
                }
            }

            device.synchronize()
                .map_err(|e| MahoutError::Cuda(format!("Sync failed: {:?}", e)))?;

            Ok(state_vector)
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda("CUDA unavailable (non-Linux)".to_string()))
        }
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
                let error_msg = if ret == 2 {
                    format!(
                        "Kernel launch reported cudaErrorMemoryAllocation (likely OOM) while encoding chunk starting at offset {} (len={}).",
                        chunk_offset,
                        chunk_len
                    )
                } else {
                    format!(
                        "Kernel launch failed with CUDA error code: {} ({})",
                        ret,
                        cuda_error_to_string(ret)
                    )
                };
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
