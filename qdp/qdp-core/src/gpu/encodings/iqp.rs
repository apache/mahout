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

// IQP (Instantaneous Quantum Polynomial) encoding
//
// Creates quantum states using diagonal unitary circuits:
// |ψ(x)⟩ = H^⊗n U_diag(x) H^⊗n |0⟩^n = 1/√(2^n) Σ_z exp(i·φ(z,x)) |z⟩
//
// Supports three entanglement patterns:
// - None: Single-qubit rotations only (φ = Σ_i x_i·z_i)
// - Linear: Nearest-neighbor (φ = Σ_i x_i·z_i + Σ_i x_i·x_{i+1}·z_i·z_{i+1})
// - Full: All-pairs (φ = Σ_i x_i·z_i + Σ_{i<j} x_i·x_j·z_i·z_j)

use super::QuantumEncoder;
#[cfg(target_os = "linux")]
use crate::error::cuda_error_to_string;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::gpu::memory::{ensure_device_memory_available, map_allocation_error};
#[cfg(target_os = "linux")]
use cudarc::driver::DevicePtr;
#[cfg(target_os = "linux")]
use std::ffi::c_void;

/// Entanglement pattern for IQP encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)] // Explicit repr for C ABI compatibility with CUDA enum
pub enum IqpEntanglement {
    /// No entanglement: single-qubit rotations only
    /// Fastest option, O(n) phase computation per amplitude
    #[default]
    None = 0,

    /// Linear entanglement: nearest-neighbor interactions
    /// Moderate expressivity with O(n) phase computation
    Linear = 1,

    /// Full entanglement: all-pairs interactions
    /// Maximum expressivity with O(n²) phase computation
    Full = 2,
}

impl IqpEntanglement {
    /// Convert to C enum value
    pub fn to_c_int(self) -> i32 {
        self as i32
    }
}

/// IQP (Instantaneous Quantum Polynomial) encoder
///
/// Creates quantum states using diagonal unitary circuits, directly computing
/// amplitudes without circuit simulation. Each basis state |z⟩ gets amplitude:
///
/// ψ_z = 1/√(2^n) · exp(i·φ(z,x))
///
/// The phase function φ depends on the entanglement pattern:
/// - None: φ(z,x) = Σ_i x_i·z_i
/// - Linear: φ(z,x) = Σ_i x_i·z_i + Σ_i x_i·x_{i+1}·z_i·z_{i+1}
/// - Full: φ(z,x) = Σ_i x_i·z_i + Σ_{i<j} x_i·x_j·z_i·z_j
///
/// # Example
/// ```ignore
/// use qdp_core::gpu::encodings::iqp::{IqpEncoder, IqpEntanglement};
///
/// let encoder = IqpEncoder::new(IqpEntanglement::Linear);
/// let data = vec![0.5, 1.0, 0.25]; // 3 features
/// let state = encoder.encode(&device, &data, 3)?; // 3 qubits -> 8 amplitudes
/// ```
pub struct IqpEncoder {
    entanglement: IqpEntanglement,
}

impl IqpEncoder {
    /// Create a new IQP encoder with the specified entanglement pattern
    pub fn new(entanglement: IqpEntanglement) -> Self {
        Self { entanglement }
    }

    /// Create an IQP encoder with no entanglement (default)
    pub fn no_entanglement() -> Self {
        Self::new(IqpEntanglement::None)
    }

    /// Create an IQP encoder with linear (nearest-neighbor) entanglement
    pub fn linear() -> Self {
        Self::new(IqpEntanglement::Linear)
    }

    /// Create an IQP encoder with full (all-pairs) entanglement
    pub fn full() -> Self {
        Self::new(IqpEntanglement::Full)
    }

    /// Get the entanglement pattern
    pub fn entanglement(&self) -> IqpEntanglement {
        self.entanglement
    }
}

impl Default for IqpEncoder {
    fn default() -> Self {
        Self::new(IqpEntanglement::default())
    }
}

impl QuantumEncoder for IqpEncoder {
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
            // Pre-flight memory check for input buffer
            let input_bytes = std::mem::size_of_val(data);
            ensure_device_memory_available(input_bytes, "IQP input buffer", Some(num_qubits))?;

            // Allocate GPU state vector
            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new(device, num_qubits)?
            };

            // Upload input data to GPU
            let input_gpu = {
                crate::profile_scope!("GPU::H2D_Input");
                device.htod_sync_copy(data).map_err(|e| {
                    map_allocation_error(input_bytes, "IQP input upload", Some(num_qubits), e)
                })?
            };

            let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
                MahoutError::InvalidInput(
                    "State vector precision mismatch (expected float64 buffer)".to_string(),
                )
            })?;

            // Launch IQP encoding kernel
            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    qdp_kernels::launch_iqp_encode(
                        *input_gpu.device_ptr() as *const f64,
                        state_ptr as *mut c_void,
                        data.len(),
                        num_qubits,
                        state_len,
                        self.entanglement.to_c_int(),
                        std::ptr::null_mut(), // default stream
                    )
                }
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "IQP encoding kernel failed with CUDA error code: {} ({})",
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
        crate::profile_scope!("IqpEncoder::encode_batch");

        // Validate inputs
        if num_samples == 0 {
            return Err(MahoutError::InvalidInput(
                "Number of samples must be at least 1".to_string(),
            ));
        }

        if sample_size == 0 {
            return Err(MahoutError::InvalidInput(
                "Sample size must be at least 1".to_string(),
            ));
        }

        if num_qubits == 0 || num_qubits > 30 {
            return Err(MahoutError::InvalidInput(format!(
                "Number of qubits {} must be between 1 and 30",
                num_qubits
            )));
        }

        // Use checked multiplication to prevent overflow
        let expected_len = num_samples.checked_mul(sample_size).ok_or_else(|| {
            MahoutError::InvalidInput(
                "Batch size overflow: num_samples * sample_size exceeds usize::MAX".to_string(),
            )
        })?;
        if batch_data.len() != expected_len {
            return Err(MahoutError::InvalidInput(format!(
                "Batch data length {} doesn't match num_samples * sample_size = {}",
                batch_data.len(),
                expected_len
            )));
        }

        // Validate batch data for NaN/Inf values (consistent with single encode)
        for (i, &val) in batch_data.iter().enumerate() {
            if !val.is_finite() {
                return Err(MahoutError::InvalidInput(format!(
                    "Batch data contains non-finite value at index {}: {}",
                    i, val
                )));
            }
        }

        let state_len = 1 << num_qubits;

        // Pre-flight memory check for input buffer
        let input_bytes = std::mem::size_of_val(batch_data);
        ensure_device_memory_available(input_bytes, "IQP batch input buffer", Some(num_qubits))?;

        // Allocate batch state vector
        let batch_state_vector = {
            crate::profile_scope!("GPU::AllocBatch");
            GpuStateVector::new_batch(device, num_samples, num_qubits)?
        };

        // Upload input data to GPU
        let input_batch_gpu = {
            crate::profile_scope!("GPU::H2D_InputBatch");
            device.htod_sync_copy(batch_data).map_err(|e| {
                map_allocation_error(input_bytes, "IQP batch input upload", Some(num_qubits), e)
            })?
        };

        let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
            MahoutError::InvalidInput(
                "Batch state vector precision mismatch (expected float64 buffer)".to_string(),
            )
        })?;

        // Launch batch IQP kernel
        {
            crate::profile_scope!("GPU::BatchKernelLaunch");
            let ret = unsafe {
                qdp_kernels::launch_iqp_encode_batch(
                    *input_batch_gpu.device_ptr() as *const f64,
                    state_ptr as *mut c_void,
                    num_samples,
                    sample_size,
                    num_qubits,
                    state_len,
                    self.entanglement.to_c_int(),
                    std::ptr::null_mut(), // default stream
                )
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!(
                    "Batch IQP encoding kernel failed: {} ({})",
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
        if data.is_empty() {
            return Err(MahoutError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }
        // Check for NaN/Inf values
        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(MahoutError::InvalidInput(format!(
                    "Input data contains non-finite value at index {}: {}",
                    i, val
                )));
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "iqp"
    }

    fn description(&self) -> &'static str {
        match self.entanglement {
            IqpEntanglement::None => "IQP encoding: diagonal unitary circuit (no entanglement)",
            IqpEntanglement::Linear => {
                "IQP encoding: diagonal unitary circuit (linear entanglement)"
            }
            IqpEntanglement::Full => "IQP encoding: diagonal unitary circuit (full entanglement)",
        }
    }
}
