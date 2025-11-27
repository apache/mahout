pub mod dlpack;
pub mod gpu;
pub mod error;

pub use error::{MahoutError, Result};

use std::sync::Arc;
use cudarc::driver::CudaDevice;
use crate::dlpack::DLManagedTensor;
use crate::gpu::{get_encoder, GpuStateVector};

/// Main entry point for Mahout QDP
/// 
/// Manages GPU context and dispatches encoding tasks.
/// Provides unified interface for device management, memory allocation, and DLPack.
pub struct QdpEngine {
    device: Arc<CudaDevice>,
}

impl QdpEngine {
    /// Initialize engine on GPU device
    /// 
    /// # Arguments
    /// * `device_id` - CUDA device ID (typically 0)
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| MahoutError::Cuda(format!("Failed to initialize CUDA device {}: {:?}", device_id, e)))?;
        Ok(Self { 
            device  // CudaDevice::new already returns Arc<CudaDevice> in cudarc 0.11
        })
    }

    /// Encode classical data into quantum state
    /// 
    /// Selects encoding strategy, executes on GPU, returns DLPack pointer.
    /// 
    /// # Arguments
    /// * `data` - Input data
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    /// 
    /// # Returns
    /// DLPack pointer for zero-copy PyTorch integration
    /// 
    /// # Safety
    /// Pointer freed by DLPack deleter, do not free manually.
    pub fn encode(
        &self,
        data: &[f64],
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        let encoder = get_encoder(encoding_method)?;
        let state_vector = encoder.encode(&self.device, data, num_qubits)?;
        Ok(state_vector.to_dlpack())
    }

    /// Get CUDA device reference for advanced operations
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

// Re-export key types for convenience
pub use gpu::QuantumEncoder;
