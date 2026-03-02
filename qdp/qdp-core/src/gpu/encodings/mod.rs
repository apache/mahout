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

// Quantum encoding strategies (Strategy Pattern)

use std::sync::Arc;

use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;
use crate::preprocessing::Preprocessor;
use cudarc::driver::CudaDevice;
#[cfg(target_os = "linux")]
use std::ffi::c_void;

/// Maximum number of qubits supported (16GB GPU memory limit)
/// This constant must match MAX_QUBITS in qdp-kernels/src/kernel_config.h
pub const MAX_QUBITS: usize = 30;

/// Validates qubit count against practical limits.
///
/// Checks:
/// - Qubit count is at least 1
/// - Qubit count does not exceed MAX_QUBITS
///
/// # Arguments
/// * `num_qubits` - The number of qubits to validate
///
/// # Returns
/// * `Ok(())` if the qubit count is valid
/// * `Err(MahoutError::InvalidInput)` if the qubit count is invalid
pub fn validate_qubit_count(num_qubits: usize) -> Result<()> {
    if num_qubits == 0 {
        return Err(MahoutError::InvalidInput(
            "Number of qubits must be at least 1".to_string(),
        ));
    }
    if num_qubits > MAX_QUBITS {
        return Err(MahoutError::InvalidInput(format!(
            "Number of qubits {} exceeds practical limit of {}",
            num_qubits, MAX_QUBITS
        )));
    }
    Ok(())
}

/// Quantum encoding strategy interface
/// Implemented by: AmplitudeEncoder, AngleEncoder, BasisEncoder
pub trait QuantumEncoder: Send + Sync {
    /// Encode classical data to quantum state on GPU
    fn encode(
        &self,
        device: &Arc<CudaDevice>,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector>;

    /// Encode multiple samples in a single GPU allocation and kernel launch (Batch Encoding)
    fn encode_batch(
        &self,
        _device: &Arc<CudaDevice>,
        _batch_data: &[f64],
        _num_samples: usize,
        _sample_size: usize,
        _num_qubits: usize,
    ) -> Result<GpuStateVector> {
        Err(crate::error::MahoutError::NotImplemented(format!(
            "Batch encoding not implemented for {}",
            self.name()
        )))
    }

    /// Validate input data before encoding
    fn validate_input(&self, data: &[f64], num_qubits: usize) -> Result<()> {
        Preprocessor::validate_input(data, num_qubits)
    }

    /// Strategy name
    fn name(&self) -> &'static str;

    /// Strategy description
    fn description(&self) -> &'static str;

    /// Encode from existing GPU pointer (zero-copy). Default: not supported.
    ///
    /// # Safety
    /// Caller must ensure `input_d` points to valid GPU memory with at least `input_len` elements
    /// of the expected dtype on the same device as `device`, and `stream` is a valid CUDA stream or null.
    #[cfg(target_os = "linux")]
    unsafe fn encode_from_gpu_ptr(
        &self,
        _device: &Arc<CudaDevice>,
        _input_d: *const c_void,
        _input_len: usize,
        _num_qubits: usize,
        _stream: *mut c_void,
    ) -> Result<GpuStateVector> {
        Err(MahoutError::NotImplemented(format!(
            "encode_from_gpu_ptr not supported for {}",
            self.name()
        )))
    }

    /// Encode batch from existing GPU pointer (zero-copy). Default: not supported.
    ///
    /// # Safety
    /// Caller must ensure `input_batch_d` points to valid GPU memory with at least
    /// `num_samples * sample_size` elements of the expected dtype on the same device as `device`,
    /// and `stream` is a valid CUDA stream or null.
    #[cfg(target_os = "linux")]
    unsafe fn encode_batch_from_gpu_ptr(
        &self,
        _device: &Arc<CudaDevice>,
        _input_batch_d: *const c_void,
        _num_samples: usize,
        _sample_size: usize,
        _num_qubits: usize,
        _stream: *mut c_void,
    ) -> Result<GpuStateVector> {
        Err(MahoutError::NotImplemented(format!(
            "encode_batch_from_gpu_ptr not supported for {}",
            self.name()
        )))
    }
}

// Encoding implementations
pub mod amplitude;
pub mod angle;
pub mod basis;
pub mod iqp;

pub use amplitude::AmplitudeEncoder;
pub use angle::AngleEncoder;
pub use basis::BasisEncoder;
pub use iqp::IqpEncoder;

/// Create encoder by name: "amplitude", "angle", "basis", "iqp", or "iqp-z"
pub fn get_encoder(name: &str) -> Result<Box<dyn QuantumEncoder>> {
    match name.to_lowercase().as_str() {
        "amplitude" => Ok(Box::new(AmplitudeEncoder)),
        "angle" => Ok(Box::new(AngleEncoder)),
        "basis" => Ok(Box::new(BasisEncoder)),
        "iqp" => Ok(Box::new(IqpEncoder::full())),
        "iqp-z" => Ok(Box::new(IqpEncoder::z_only())),
        _ => Err(crate::error::MahoutError::InvalidInput(format!(
            "Unknown encoder: {}. Available: amplitude, angle, basis, iqp, iqp-z",
            name
        ))),
    }
}
