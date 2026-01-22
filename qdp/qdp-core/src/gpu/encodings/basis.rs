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

// Basis encoding: map integers to computational basis states

use super::{QuantumEncoder, validate_qubit_count};
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

/// Basis encoding: maps an integer index to a computational basis state.
///
/// For n qubits, maps integer i (0 ≤ i < 2^n) to |i⟩, where:
/// - state[i] = 1.0 + 0.0i
/// - state[j] = 0.0 + 0.0i for all j ≠ i
///
/// Example: index 3 with 3 qubits → |011⟩ (binary representation of 3)
///
/// Input format:
/// - Single encoding: data = [index] (single f64 representing the basis index)
/// - Batch encoding: data = [idx0, idx1, ..., idxN] (one index per sample)
pub struct BasisEncoder;

impl QuantumEncoder for BasisEncoder {
    fn encode(
        &self,
        #[cfg(target_os = "linux")] device: &Arc<CudaDevice>,
        #[cfg(not(target_os = "linux"))] _device: &Arc<CudaDevice>,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        // Validate basic input constraints
        self.validate_input(data, num_qubits)?;

        // For basis encoding, we expect exactly one value: the basis index
        if data.len() != 1 {
            return Err(MahoutError::InvalidInput(format!(
                "Basis encoding expects exactly 1 value (the basis index), got {}",
                data.len()
            )));
        }

        let state_len = 1 << num_qubits;

        #[cfg(target_os = "linux")]
        {
            // Convert and validate the basis index
            let basis_index = Self::validate_basis_index(data[0], state_len)?;
            // Allocate GPU state vector
            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new(device, num_qubits)?
            };

            let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
                MahoutError::InvalidInput(
                    "State vector precision mismatch (expected float64 buffer)".to_string(),
                )
            })?;

            // Launch basis encoding kernel
            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    qdp_kernels::launch_basis_encode(
                        basis_index,
                        state_ptr as *mut c_void,
                        state_len,
                        std::ptr::null_mut(), // default stream
                    )
                }
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Basis encoding kernel failed with CUDA error code: {} ({})",
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

    /// Encode multiple basis indices in a single GPU allocation and kernel launch
    #[cfg(target_os = "linux")]
    fn encode_batch(
        &self,
        device: &Arc<CudaDevice>,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        crate::profile_scope!("BasisEncoder::encode_batch");

        // For basis encoding, each sample should have exactly 1 value (the index)
        if sample_size != 1 {
            return Err(MahoutError::InvalidInput(format!(
                "Basis encoding expects sample_size=1 (one index per sample), got {}",
                sample_size
            )));
        }

        if batch_data.len() != num_samples {
            return Err(MahoutError::InvalidInput(format!(
                "Batch data length {} doesn't match num_samples {}",
                batch_data.len(),
                num_samples
            )));
        }

        validate_qubit_count(num_qubits)?;

        let state_len = 1 << num_qubits;

        // Convert and validate all basis indices
        let basis_indices: Vec<usize> = batch_data
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                Self::validate_basis_index(val, state_len)
                    .map_err(|e| MahoutError::InvalidInput(format!("Sample {}: {}", i, e)))
            })
            .collect::<Result<Vec<_>>>()?;

        // Allocate batch state vector
        let batch_state_vector = {
            crate::profile_scope!("GPU::AllocBatch");
            GpuStateVector::new_batch(device, num_samples, num_qubits)?
        };

        // Upload basis indices to GPU
        let indices_gpu = {
            crate::profile_scope!("GPU::H2D_Indices");
            device.htod_sync_copy(&basis_indices).map_err(|e| {
                map_allocation_error(
                    num_samples * std::mem::size_of::<usize>(),
                    "basis indices upload",
                    Some(num_qubits),
                    e,
                )
            })?
        };

        let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "Batch state vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        // Launch batch kernel
        {
            crate::profile_scope!("GPU::BatchKernelLaunch");
            let ret = unsafe {
                qdp_kernels::launch_basis_encode_batch(
                    *indices_gpu.device_ptr() as *const usize,
                    state_ptr as *mut c_void,
                    num_samples,
                    state_len,
                    num_qubits as u32,
                    std::ptr::null_mut(), // default stream
                )
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Batch basis encoding kernel failed: {} ({})",
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

    fn validate_input(&self, data: &[f64], num_qubits: usize) -> Result<()> {
        // Basic validation: qubits and data availability
        validate_qubit_count(num_qubits)?;
        if data.is_empty() {
            return Err(MahoutError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "basis"
    }

    fn description(&self) -> &'static str {
        "Basis encoding: maps integers to computational basis states"
    }
}

impl BasisEncoder {
    /// Validate and convert a f64 value to a valid basis index
    fn validate_basis_index(value: f64, state_len: usize) -> Result<usize> {
        // Check for non-finite values
        if !value.is_finite() {
            return Err(MahoutError::InvalidInput(
                "Basis index must be a finite number".to_string(),
            ));
        }

        // Check for negative values
        if value < 0.0 {
            return Err(MahoutError::InvalidInput(format!(
                "Basis index must be non-negative, got {}",
                value
            )));
        }

        // Check if the value is an integer
        if value.fract() != 0.0 {
            return Err(MahoutError::InvalidInput(format!(
                "Basis index must be an integer, got {} (hint: use .round() if needed)",
                value
            )));
        }

        // Convert to usize
        let index = value as usize;

        // Check bounds
        if index >= state_len {
            return Err(MahoutError::InvalidInput(format!(
                "Basis index {} exceeds state vector size {} (max index: {})",
                index,
                state_len,
                state_len - 1
            )));
        }

        Ok(index)
    }
}
