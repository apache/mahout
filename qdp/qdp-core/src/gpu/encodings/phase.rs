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

// Phase encoding: map per-qubit phase angles to a product state.
//
// Each qubit k is initialized via H|0⟩ = |+⟩ = (1/√2)(|0⟩ + |1⟩), then
// a phase gate P(φ = x_k) is applied, giving:
//
//   |x̄_k⟩ = P(φ = x_k)|+⟩ = (1/√2)(|0⟩ + e^{i x_k}|1⟩)
//
// The full N-qubit state is the tensor product of these single-qubit states:
//
//   |x̄⟩ = ⊗_{k=1}^{N} (1/√2)(|0⟩ + e^{i x_k}|1⟩)
//         = (1/√2^N) Σ_{b=0}^{2^N-1}  e^{i Σ_k x_k · b_k}  |b⟩
//
// where b_k is the k-th bit of basis index b.  Input x_k ∈ (0, 2π] is
// recommended to avoid information loss due to 2π periodicity of e^{iφ}.
//
// Circuit depth: 2 (one Hadamard layer + one phase-gate layer).

#![allow(unused_unsafe)]

use super::{QuantumEncoder, validate_qubit_count};
#[cfg(target_os = "linux")]
use crate::error::cuda_error_to_string;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::{GpuStateVector, Precision};
#[cfg(target_os = "linux")]
use crate::gpu::pipeline::run_dual_stream_pipeline_aligned;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::gpu::memory::map_allocation_error;
#[cfg(target_os = "linux")]
use cudarc::driver::DevicePtr;
#[cfg(target_os = "linux")]
use std::ffi::c_void;

/// Phase encoding: per-qubit P(φ = x_k) gates applied to |+⟩^⊗N.
///
/// Input:  `data` — n f64 values x_k ∈ (0, 2π], one phase angle per qubit.
///         Values outside (0, 2π] are accepted but may cause information loss
///         due to 2π periodicity of e^{iφ}.
/// Output: `GpuStateVector` of length 2^n encoding the product state
///         (1/√2^n) Σ_b e^{i Σ_k x_k · b_k} |b⟩.
///
/// Kernel contract: `launch_phase_encode(phases, state, state_len, num_qubits, stream)`
/// must write amplitude[b] = (1/√2^n) · e^{i Σ_k phases[k] · ((b >> k) & 1)}
/// as an interleaved (re, im) f64 pair for each basis index b.
pub struct PhaseEncoder;

impl QuantumEncoder for PhaseEncoder {
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
            let phases_gpu = {
                crate::profile_scope!("GPU::H2D_Phases");
                device.htod_sync_copy(data).map_err(|e| {
                    map_allocation_error(
                        input_bytes,
                        "phase input upload",
                        Some(num_qubits),
                        Some(device.ordinal()),
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

            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    qdp_kernels::launch_phase_encode(
                        *phases_gpu.device_ptr() as *const f64,
                        state_ptr as *mut c_void,
                        state_len,
                        num_qubits as u32,
                        std::ptr::null_mut(),
                    )
                }
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Phase encoding kernel failed with CUDA error code: {} ({})",
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

    /// Encode multiple phase samples in a single GPU allocation and kernel launch.
    #[cfg(target_os = "linux")]
    fn encode_batch(
        &self,
        device: &Arc<CudaDevice>,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        crate::profile_scope!("PhaseEncoder::encode_batch");

        // Each sample provides exactly one phase angle per qubit.
        if sample_size != num_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "Phase encoding expects sample_size={} (one angle per qubit), got {}",
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

        validate_qubit_count(num_qubits)?;

        for (i, &val) in batch_data.iter().enumerate() {
            if !val.is_finite() {
                let sample_idx = i / sample_size;
                let angle_idx = i % sample_size;
                return Err(MahoutError::InvalidInput(format!(
                    "Sample {} phase angle {} must be finite, got {}",
                    sample_idx, angle_idx, val
                )));
            }
        }

        let state_len = 1 << num_qubits;

        const ASYNC_THRESHOLD_ELEMENTS: usize = 1024 * 1024 / std::mem::size_of::<f64>(); // 1 MB
        if batch_data.len() >= ASYNC_THRESHOLD_ELEMENTS {
            return Self::encode_batch_async_pipeline(
                device,
                batch_data,
                num_samples,
                sample_size,
                num_qubits,
                state_len,
            );
        }

        let batch_state_vector = {
            crate::profile_scope!("GPU::AllocBatch");
            GpuStateVector::new_batch(device, num_samples, num_qubits, Precision::Float64)?
        };

        let input_bytes = std::mem::size_of_val(batch_data);
        let phases_gpu = {
            crate::profile_scope!("GPU::H2D_BatchPhases");
            device.htod_sync_copy(batch_data).map_err(|e| {
                map_allocation_error(
                    input_bytes,
                    "phase batch upload",
                    Some(num_qubits),
                    Some(device.ordinal()),
                    e,
                )
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
                qdp_kernels::launch_phase_encode_batch(
                    *phases_gpu.device_ptr() as *const f64,
                    state_ptr as *mut c_void,
                    num_samples,
                    state_len,
                    num_qubits as u32,
                    std::ptr::null_mut(),
                )
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Batch phase encoding kernel failed: {} ({})",
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

    #[cfg(target_os = "linux")]
    unsafe fn encode_from_gpu_ptr(
        &self,
        device: &Arc<CudaDevice>,
        input_d: *const c_void,
        input_len: usize,
        num_qubits: usize,
        stream: *mut c_void,
    ) -> Result<GpuStateVector> {
        if input_len != num_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "Phase encoding expects {} values (one angle per qubit), got {}",
                num_qubits, input_len
            )));
        }
        let state_len = 1 << num_qubits;
        let phases_d = input_d as *const f64;

        let state_vector = {
            crate::profile_scope!("GPU::Alloc");
            GpuStateVector::new(device, num_qubits, Precision::Float64)?
        };
        let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "State vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        {
            crate::profile_scope!("GPU::KernelLaunch");
            let ret = unsafe {
                qdp_kernels::launch_phase_encode(
                    phases_d,
                    state_ptr as *mut c_void,
                    state_len,
                    num_qubits as u32,
                    stream,
                )
            };
            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Phase encoding kernel failed with CUDA error code: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }
        }

        {
            crate::profile_scope!("GPU::Synchronize");
            crate::gpu::cuda_sync::sync_cuda_stream(stream, "CUDA stream synchronize failed")?;
        }

        Ok(state_vector)
    }

    #[cfg(target_os = "linux")]
    unsafe fn encode_batch_from_gpu_ptr(
        &self,
        device: &Arc<CudaDevice>,
        input_batch_d: *const c_void,
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        stream: *mut c_void,
    ) -> Result<GpuStateVector> {
        if sample_size == 0 {
            return Err(MahoutError::InvalidInput(
                "Sample size cannot be zero".into(),
            ));
        }
        if sample_size != num_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "Phase encoding expects sample_size={} (one angle per qubit), got {}",
                num_qubits, sample_size
            )));
        }
        let state_len = 1 << num_qubits;
        let input_batch_d = input_batch_d as *const f64;

        // Use the L2-norm kernel as a finiteness probe: NaN/Inf propagates
        // through the norm and is caught on the host side.
        let phase_validation_buffer = {
            crate::profile_scope!("GPU::PhaseFiniteCheckBatch");
            use cudarc::driver::DevicePtrMut;
            let mut buffer = device.alloc_zeros::<f64>(num_samples).map_err(|e| {
                MahoutError::MemoryAllocation(format!(
                    "Failed to allocate phase validation buffer: {:?}",
                    e
                ))
            })?;
            let ret = unsafe {
                qdp_kernels::launch_l2_norm_batch(
                    input_batch_d,
                    num_samples,
                    sample_size,
                    *buffer.device_ptr_mut() as *mut f64,
                    stream,
                )
            };
            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Phase validation norm kernel failed with CUDA error code: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }
            buffer
        };

        {
            crate::profile_scope!("GPU::PhaseFiniteValidationHostCopy");
            let host_norms = device
                .dtoh_sync_copy(&phase_validation_buffer)
                .map_err(|e| {
                    MahoutError::Cuda(format!(
                        "Failed to copy phase validation norms to host: {:?}",
                        e
                    ))
                })?;
            if host_norms.iter().any(|v| !v.is_finite()) {
                return Err(MahoutError::InvalidInput(
                    "Phase encoding batch contains non-finite values (NaN or Inf)".to_string(),
                ));
            }
        }

        let batch_state_vector = {
            crate::profile_scope!("GPU::AllocBatch");
            GpuStateVector::new_batch(device, num_samples, num_qubits, Precision::Float64)?
        };
        let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "Batch state vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        {
            crate::profile_scope!("GPU::BatchKernelLaunch");
            let ret = unsafe {
                qdp_kernels::launch_phase_encode_batch(
                    input_batch_d,
                    state_ptr as *mut c_void,
                    num_samples,
                    state_len,
                    num_qubits as u32,
                    stream,
                )
            };
            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Batch phase encoding kernel failed: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                )));
            }
        }

        {
            crate::profile_scope!("GPU::Synchronize");
            crate::gpu::cuda_sync::sync_cuda_stream(stream, "CUDA stream synchronize failed")?;
        }

        Ok(batch_state_vector)
    }

    fn validate_input(&self, data: &[f64], num_qubits: usize) -> Result<()> {
        validate_qubit_count(num_qubits)?;
        if data.len() != num_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "Phase encoding expects {} values (one angle per qubit), got {}",
                num_qubits,
                data.len()
            )));
        }
        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(MahoutError::InvalidInput(format!(
                    "Phase angle at index {} must be finite, got {}",
                    i, val
                )));
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "phase"
    }

    fn description(&self) -> &'static str {
        "Phase encoding: per-qubit P(φ=x_k) gates on |+⟩, producing ⊗_k (1/√2)(|0⟩ + e^{ix_k}|1⟩)"
    }
}

impl PhaseEncoder {
    #[cfg(target_os = "linux")]
    fn encode_batch_async_pipeline(
        device: &Arc<CudaDevice>,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        state_len: usize,
    ) -> Result<GpuStateVector> {
        let batch_state_vector = {
            crate::profile_scope!("GPU::AllocBatch");
            GpuStateVector::new_batch(device, num_samples, num_qubits, Precision::Float64)?
        };

        let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "Batch state vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        run_dual_stream_pipeline_aligned(
            device,
            batch_data,
            sample_size,
            |stream, input_ptr, chunk_offset, chunk_len| {
                if chunk_len % sample_size != 0 || chunk_offset % sample_size != 0 {
                    return Err(MahoutError::InvalidInput(
                        "Phase batch chunk is not aligned to sample size".to_string(),
                    ));
                }

                let chunk_samples = chunk_len / sample_size;
                let sample_offset = chunk_offset / sample_size;
                let offset_elements = sample_offset.checked_mul(state_len).ok_or_else(|| {
                    MahoutError::InvalidInput("Phase batch output offset overflow".to_string())
                })?;

                let state_ptr_offset = unsafe { state_ptr.add(offset_elements) as *mut c_void };
                let ret = unsafe {
                    qdp_kernels::launch_phase_encode_batch(
                        input_ptr,
                        state_ptr_offset,
                        chunk_samples,
                        state_len,
                        num_qubits as u32,
                        stream.stream as *mut c_void,
                    )
                };

                if ret != 0 {
                    return Err(MahoutError::KernelLaunch(format!(
                        "Batch phase encoding kernel failed: {} ({})",
                        ret,
                        cuda_error_to_string(ret)
                    )));
                }

                Ok(())
            },
        )?;

        Ok(batch_state_vector)
    }
}
