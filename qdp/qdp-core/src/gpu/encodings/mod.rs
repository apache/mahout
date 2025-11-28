// Quantum encoding strategies (Strategy Pattern)

use std::sync::Arc;
use cudarc::driver::CudaDevice;
use crate::error::Result;
use crate::gpu::memory::GpuStateVector;

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

    /// Strategy name
    fn name(&self) -> &'static str;

    /// Strategy description
    fn description(&self) -> &'static str;
}

// Encoding implementations
pub mod amplitude;
pub mod angle;
pub mod basis;

pub use amplitude::AmplitudeEncoder;
pub use angle::AngleEncoder;
pub use basis::BasisEncoder;

/// Create encoder by name: "amplitude", "angle", or "basis"
pub fn get_encoder(name: &str) -> Result<Box<dyn QuantumEncoder>> {
    match name.to_lowercase().as_str() {
        "amplitude" => Ok(Box::new(AmplitudeEncoder)),
        "angle" => Ok(Box::new(AngleEncoder)),
        "basis" => Ok(Box::new(BasisEncoder)),
        _ => Err(crate::error::MahoutError::InvalidInput(
            format!("Unknown encoder: {}. Available: amplitude, angle, basis", name)
        )),
    }
}

