use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use qdp_core::QdpEngine as CoreEngine;

/// PyO3 wrapper for QdpEngine
///
/// Provides Python bindings for GPU-accelerated quantum state encoding.
#[pyclass]
struct QdpEngine {
    engine: CoreEngine,
}

#[pymethods]
impl QdpEngine {
    /// Initialize QDP engine on specified GPU device
    ///
    /// Args:
    ///     device_id: CUDA device ID (typically 0)
    ///
    /// Returns:
    ///     QdpEngine instance
    ///
    /// Raises:
    ///     RuntimeError: If CUDA device initialization fails
    #[new]
    fn new(device_id: usize) -> PyResult<Self> {
        let engine = CoreEngine::new(device_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;
        Ok(Self { engine })
    }

    /// Encode classical data into quantum state
    ///
    /// Args:
    ///     data: Input data as list of floats
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy ("amplitude", "angle", or "basis")
    ///
    /// Returns:
    ///     DLPack tensor pointer (integer) for zero-copy PyTorch integration
    ///
    /// Raises:
    ///     RuntimeError: If encoding fails
    fn encode(&self, data: Vec<f64>, num_qubits: usize, encoding_method: &str) -> PyResult<usize> {
        let ptr = self.engine.encode(&data, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
        Ok(ptr as usize)
    }
}

/// Mahout QDP Python module
///
/// GPU-accelerated quantum data encoding with DLPack integration.
#[pymodule]
fn mahout_qdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QdpEngine>()?;
    Ok(())
}
