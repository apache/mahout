use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use qdp_core::QdpEngine as CoreEngine;
use qdp_core::dlpack::DLManagedTensor;

/// Quantum tensor wrapper implementing DLPack protocol
///
/// This class wraps a GPU-allocated quantum state vector and implements
/// the DLPack protocol for zero-copy integration with PyTorch and other
/// array libraries.
///
/// Example:
///     >>> engine = QdpEngine(device_id=0)
///     >>> qtensor = engine.encode([1.0, 2.0, 3.0], num_qubits=2, encoding_method="amplitude")
///     >>> torch_tensor = torch.from_dlpack(qtensor)
#[pyclass]
struct QuantumTensor {
    ptr: *mut DLManagedTensor,
    consumed: bool,
}

#[pymethods]
impl QuantumTensor {
    /// Implements DLPack protocol - returns PyCapsule for PyTorch
    ///
    /// This method is called by torch.from_dlpack() to get the GPU memory pointer.
    /// The capsule can only be consumed once to prevent double-free errors.
    ///
    /// Args:
    ///     stream: Optional CUDA stream pointer (for DLPack 0.8+)
    ///
    /// Returns:
    ///     PyCapsule containing DLManagedTensor pointer
    ///
    /// Raises:
    ///     RuntimeError: If the tensor has already been consumed
    #[pyo3(signature = (stream=None))]
    fn __dlpack__<'py>(&mut self, py: Python<'py>, stream: Option<i64>) -> PyResult<Py<PyAny>> {
        let _ = stream;  // Suppress unused variable warning
        if self.consumed {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor already consumed (can only be used once)"
            ));
        }

        if self.ptr.is_null() {
            return Err(PyRuntimeError::new_err("Invalid DLPack tensor pointer"));
        }

        // Mark as consumed to prevent double-free
        self.consumed = true;

        // Create PyCapsule using FFI
        // PyTorch will call the deleter stored in DLManagedTensor.deleter
        // Use a static C string for the capsule name to avoid lifetime issues
        const DLTENSOR_NAME: &[u8] = b"dltensor\0";

        unsafe {
            // Create PyCapsule without a destructor
            // PyTorch will manually call the deleter from DLManagedTensor
            let capsule_ptr = ffi::PyCapsule_New(
                self.ptr as *mut std::ffi::c_void,
                DLTENSOR_NAME.as_ptr() as *const i8,
                None  // No destructor - PyTorch handles it
            );

            if capsule_ptr.is_null() {
                return Err(PyRuntimeError::new_err("Failed to create PyCapsule"));
            }

            Ok(Py::from_owned_ptr(py, capsule_ptr))
        }
    }

    /// Returns DLPack device information
    ///
    /// Returns:
    ///     Tuple of (device_type, device_id) where device_type=2 for CUDA
    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        // DLDeviceType::kDLCUDA = 2, device_id = 0
        Ok((2, 0))
    }
}

impl Drop for QuantumTensor {
    fn drop(&mut self) {
        // Only free if not consumed by __dlpack__
        // If consumed, PyTorch/consumer will call the deleter
        if !self.consumed && !self.ptr.is_null() {
            unsafe {
                // Call the DLPack deleter to properly free memory
                if let Some(deleter) = (*self.ptr).deleter {
                    deleter(self.ptr);
                }
            }
        }
    }
}

// Safety: QuantumTensor can be sent between threads
// The DLManagedTensor pointer management is thread-safe via Arc in the deleter
unsafe impl Send for QuantumTensor {}
unsafe impl Sync for QuantumTensor {}

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
    ///     QuantumTensor: DLPack-compatible tensor for zero-copy PyTorch integration
    ///
    /// Raises:
    ///     RuntimeError: If encoding fails
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> qtensor = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2, encoding_method="amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(qtensor)
    ///
    /// TODO: Replace Vec<f64> with numpy array input to enable zero-copy reading.
    /// Consider using the numpy crate (e.g., PyReadonlyArray1<f64>) for better performance.
    fn encode(&self, data: Vec<f64>, num_qubits: usize, encoding_method: &str) -> PyResult<QuantumTensor> {
        let ptr = self.engine.encode(&data, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }
}

/// Mahout QDP Python module
///
/// GPU-accelerated quantum data encoding with DLPack integration.
#[pymodule]
fn mahout_qdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QdpEngine>()?;
    m.add_class::<QuantumTensor>()?;
    Ok(())
}
