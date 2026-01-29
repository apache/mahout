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

// Amplitude encoding: state injection with L2 normalization

// Allow unused_unsafe: qdp_kernels functions are unsafe in CUDA builds but safe stubs in no-CUDA builds.
// The compiler can't statically determine which path is taken.
#![allow(unused_unsafe)]

use std::sync::Arc;

use super::QuantumEncoder;
#[cfg(target_os = "linux")]
use crate::error::cuda_error_to_string;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;
use crate::gpu::pipeline::run_dual_stream_pipeline;
use cudarc::driver::CudaDevice;

#[cfg(target_os = "linux")]
use crate::gpu::cuda_ffi::cudaMemsetAsync;
#[cfg(target_os = "linux")]
use crate::gpu::memory::{ensure_device_memory_available, map_allocation_error};
#[cfg(target_os = "linux")]
use cudarc::driver::{DevicePtr, DevicePtrMut};
#[cfg(target_os = "linux")]
use qdp_kernels::{
    launch_amplitude_encode, launch_amplitude_encode_batch, launch_l2_norm, launch_l2_norm_batch,
    launch_l2_norm_f32,
};
#[cfg(target_os = "linux")]
use std::ffi::c_void;

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
        // Validate qubits using Preprocessor (which uses validate_qubit_count internally)
        Preprocessor::validate_input(host_data, num_qubits)?;
        let state_len = 1 << num_qubits;

        #[cfg(target_os = "linux")]
        {
            // Allocate GPU state vector
            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new(_device, num_qubits)?
            };

            // Async Pipeline for large data
            // For small data (< 1MB), use synchronous path to avoid stream overhead
            // For large data, use dual-stream async pipeline for maximum throughput
            const ASYNC_THRESHOLD: usize = 1024 * 1024 / std::mem::size_of::<f64>(); // 1MB threshold
            const GPU_NORM_THRESHOLD: usize = 4096; // heuristic: amortize kernel launch

            if host_data.len() < ASYNC_THRESHOLD {
                // Synchronous path for small data (avoids stream overhead)
                let input_bytes = std::mem::size_of_val(host_data);
                ensure_device_memory_available(
                    input_bytes,
                    "input staging buffer",
                    Some(num_qubits),
                )?;

                let input_slice = {
                    crate::profile_scope!("GPU::H2DCopy");
                    _device.htod_sync_copy(host_data).map_err(|e| {
                        map_allocation_error(
                            input_bytes,
                            "input staging buffer",
                            Some(num_qubits),
                            e,
                        )
                    })?
                };

                // GPU-accelerated norm for medium+ inputs, CPU fallback for tiny payloads
                let inv_norm = if host_data.len() >= GPU_NORM_THRESHOLD {
                    // SAFETY: input_slice was just allocated and copied from host_data,
                    // so the pointer is valid and contains host_data.len() elements
                    unsafe {
                        Self::calculate_inv_norm_gpu(
                            _device,
                            *input_slice.device_ptr() as *const f64,
                            host_data.len(),
                        )?
                    }
                } else {
                    let norm = Preprocessor::calculate_l2_norm(host_data)?;
                    1.0 / norm
                };

                let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
                    let actual = state_vector.precision();
                    MahoutError::InvalidInput(format!(
                        "State vector precision mismatch (expected float64 buffer, got {:?})",
                        actual
                    ))
                })?;

                let ret = {
                    crate::profile_scope!("GPU::KernelLaunch");
                    unsafe {
                        launch_amplitude_encode(
                            *input_slice.device_ptr() as *const f64,
                            state_ptr as *mut c_void,
                            host_data.len(),
                            state_len,
                            inv_norm,
                            std::ptr::null_mut(), // default stream
                        )
                    }
                };

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
                    _device.synchronize().map_err(|e| {
                        MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e))
                    })?;
                }
            } else {
                // Async Pipeline path for large data
                let norm = Preprocessor::calculate_l2_norm(host_data)?;
                let inv_norm = 1.0 / norm;
                Self::encode_async_pipeline(
                    _device,
                    host_data,
                    num_qubits,
                    state_len,
                    inv_norm,
                    &state_vector,
                )?;
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

    /// Encode multiple samples in a single GPU allocation and kernel launch
    #[cfg(target_os = "linux")]
    fn encode_batch(
        &self,
        device: &Arc<CudaDevice>,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        crate::profile_scope!("AmplitudeEncoder::encode_batch");

        // Validate inputs using shared preprocessor
        Preprocessor::validate_batch(batch_data, num_samples, sample_size, num_qubits)?;

        let state_len = 1 << num_qubits;

        // Allocate single large GPU buffer for all states
        let batch_state_vector = {
            crate::profile_scope!("GPU::AllocBatch");
            GpuStateVector::new_batch(device, num_samples, num_qubits)?
        };

        // Upload input data to GPU
        let input_batch_gpu = {
            crate::profile_scope!("GPU::H2D_InputBatch");
            device.htod_sync_copy(batch_data).map_err(|e| {
                MahoutError::MemoryAllocation(format!("Failed to upload batch input: {:?}", e))
            })?
        };

        // Compute inverse norms on GPU using warp-reduced kernel
        let inv_norms_gpu = {
            crate::profile_scope!("GPU::BatchNormKernel");
            let mut buffer = device.alloc_zeros::<f64>(num_samples).map_err(|e| {
                MahoutError::MemoryAllocation(format!("Failed to allocate norm buffer: {:?}", e))
            })?;

            let ret = unsafe {
                launch_l2_norm_batch(
                    *input_batch_gpu.device_ptr() as *const f64,
                    num_samples,
                    sample_size,
                    *buffer.device_ptr_mut() as *mut f64,
                    std::ptr::null_mut(), // default stream
                )
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Norm reduction kernel failed: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }

            buffer
        };

        // Validate norms on host to catch zero or NaN samples early
        {
            crate::profile_scope!("GPU::NormValidation");
            let host_inv_norms = device
                .dtoh_sync_copy(&inv_norms_gpu)
                .map_err(|e| MahoutError::Cuda(format!("Failed to copy norms to host: {:?}", e)))?;

            if host_inv_norms.iter().any(|v| !v.is_finite() || *v == 0.0) {
                return Err(MahoutError::InvalidInput(
                    "One or more samples have zero or invalid norm".to_string(),
                ));
            }
        }

        // Launch batch kernel
        {
            crate::profile_scope!("GPU::BatchKernelLaunch");
            let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
                MahoutError::InvalidInput(
                    "Batch state vector precision mismatch (expected float64 buffer)".to_string(),
                )
            })?;
            let ret = unsafe {
                launch_amplitude_encode_batch(
                    *input_batch_gpu.device_ptr() as *const f64,
                    state_ptr as *mut c_void,
                    *inv_norms_gpu.device_ptr() as *const f64,
                    num_samples,
                    sample_size,
                    state_len,
                    std::ptr::null_mut(), // default stream
                )
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Batch kernel launch failed: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }
        }

        // Synchronize
        {
            crate::profile_scope!("GPU::Synchronize");
            device
                .synchronize()
                .map_err(|e| MahoutError::Cuda(format!("Sync failed: {:?}", e)))?;
        }

        Ok(batch_state_vector)
    }

    fn name(&self) -> &'static str {
        "amplitude"
    }

    fn description(&self) -> &'static str {
        "Amplitude encoding with L2 normalization"
    }
}

impl AmplitudeEncoder {
    /// Async pipeline encoding for large data
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
        inv_norm: f64,
        state_vector: &GpuStateVector,
    ) -> Result<()> {
        let base_state_ptr = state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "State vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        // Use generic pipeline infrastructure
        // The closure handles amplitude-specific kernel launch logic
        run_dual_stream_pipeline(
            device,
            host_data,
            |stream, input_ptr, chunk_offset, chunk_len| {
                // Calculate offset pointer for state vector (type-safe pointer arithmetic)
                // Offset is in complex numbers (CuDoubleComplex), not f64 elements
                let state_ptr_offset = unsafe {
                    base_state_ptr
                        .cast::<u8>()
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
                        inv_norm,
                        stream.stream as *mut c_void,
                    )
                };

                if ret != 0 {
                    let error_msg = if ret == 2 {
                        format!(
                            "Kernel launch reported cudaErrorMemoryAllocation (likely OOM) while encoding chunk starting at offset {} (len={}).",
                            chunk_offset, chunk_len
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
            },
        )?;

        // CRITICAL FIX: Handle padding for uninitialized memory
        // Since we use alloc() (uninitialized), we must zero-fill any tail region
        // that wasn't written by the pipeline. This ensures correctness when
        // host_data.len() < state_len (e.g., 1000 elements in a 1024-element state).
        let data_len = host_data.len();
        if data_len < state_len {
            let padding_start = data_len;
            let padding_elements = state_len - padding_start;
            let padding_bytes =
                padding_elements * std::mem::size_of::<qdp_kernels::CuDoubleComplex>();

            // Calculate tail pointer (in complex numbers)
            let tail_ptr = unsafe { base_state_ptr.add(padding_start) as *mut c_void };

            // Zero-fill padding region using CUDA Runtime API
            // Use default stream since pipeline streams are already synchronized
            unsafe {
                let result = cudaMemsetAsync(
                    tail_ptr,
                    0,
                    padding_bytes,
                    std::ptr::null_mut(), // default stream
                );

                if result != 0 {
                    return Err(MahoutError::Cuda(format!(
                        "Failed to zero-fill padding region: {} ({})",
                        result,
                        cuda_error_to_string(result)
                    )));
                }
            }

            // Synchronize to ensure padding is complete before returning
            device
                .synchronize()
                .map_err(|e| MahoutError::Cuda(format!("Failed to sync after padding: {:?}", e)))?;
        }

        Ok(())
    }
}

impl AmplitudeEncoder {
    /// Compute inverse L2 norm on GPU using the reduction kernel.
    ///
    /// # Arguments
    /// * `device` - CUDA device reference
    /// * `input_ptr` - Device pointer to input data (f64 array on GPU)
    /// * `len` - Number of f64 elements
    ///
    /// # Returns
    /// The inverse L2 norm (1/||x||_2) of the input data
    ///
    /// # Safety
    /// The caller must ensure `input_ptr` points to valid GPU memory containing
    /// at least `len` f64 elements on the same device as `device`.
    #[cfg(target_os = "linux")]
    pub(crate) unsafe fn calculate_inv_norm_gpu(
        device: &Arc<CudaDevice>,
        input_ptr: *const f64,
        len: usize,
    ) -> Result<f64> {
        crate::profile_scope!("GPU::NormSingle");

        let mut norm_buffer = device.alloc_zeros::<f64>(1).map_err(|e| {
            MahoutError::MemoryAllocation(format!("Failed to allocate norm buffer: {:?}", e))
        })?;

        let ret = unsafe {
            launch_l2_norm(
                input_ptr,
                len,
                *norm_buffer.device_ptr_mut() as *mut f64,
                std::ptr::null_mut(), // default stream
            )
        };

        if ret != 0 {
            return Err(MahoutError::KernelLaunch(format!(
                "Norm kernel failed: {} ({})",
                ret,
                cuda_error_to_string(ret)
            )));
        }

        let inv_norm_host = device
            .dtoh_sync_copy(&norm_buffer)
            .map_err(|e| MahoutError::Cuda(format!("Failed to copy norm to host: {:?}", e)))?;

        let inv_norm = inv_norm_host.first().copied().unwrap_or(0.0);
        if inv_norm == 0.0 || !inv_norm.is_finite() {
            return Err(MahoutError::InvalidInput(
                "Input data has zero or non-finite norm (contains NaN, Inf, or all zeros)"
                    .to_string(),
            ));
        }

        Ok(inv_norm)
    }

    /// Compute inverse L2 norm on GPU for float32 input using the reduction kernel.
    ///
    /// # Arguments
    /// * `device` - CUDA device reference
    /// * `input_ptr` - Device pointer to input data (f32 array on GPU)
    /// * `len` - Number of f32 elements
    ///
    /// # Returns
    /// The inverse L2 norm (1/||x||_2) of the input data as `f32`.
    ///
    /// # Safety
    /// The caller must ensure `input_ptr` points to valid GPU memory containing
    /// at least `len` f32 elements on the same device as `device`.
    #[cfg(target_os = "linux")]
    pub unsafe fn calculate_inv_norm_gpu_f32(
        device: &Arc<CudaDevice>,
        input_ptr: *const f32,
        len: usize,
    ) -> Result<f32> {
        crate::profile_scope!("GPU::NormSingleF32");

        let mut norm_buffer = device.alloc_zeros::<f32>(1).map_err(|e| {
            MahoutError::MemoryAllocation(format!("Failed to allocate f32 norm buffer: {:?}", e))
        })?;

        let ret = unsafe {
            launch_l2_norm_f32(
                input_ptr,
                len,
                *norm_buffer.device_ptr_mut() as *mut f32,
                std::ptr::null_mut(), // default stream
            )
        };

        if ret != 0 {
            return Err(MahoutError::KernelLaunch(format!(
                "Norm kernel f32 failed: {} ({})",
                ret,
                cuda_error_to_string(ret)
            )));
        }

        let inv_norm_host = device
            .dtoh_sync_copy(&norm_buffer)
            .map_err(|e| MahoutError::Cuda(format!("Failed to copy f32 norm to host: {:?}", e)))?;

        let inv_norm = inv_norm_host.first().copied().unwrap_or(0.0);
        if inv_norm == 0.0 || !inv_norm.is_finite() {
            return Err(MahoutError::InvalidInput(
                "Input data (f32) has zero or non-finite norm (contains NaN, Inf, or all zeros)"
                    .to_string(),
            ));
        }

        Ok(inv_norm)
    }
}
