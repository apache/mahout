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

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use qdp_core::dlpack::DLManagedTensor;
use qdp_core::{Precision, QdpEngine as CoreEngine};

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
        let _ = stream; // Suppress unused variable warning
        if self.consumed {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor already consumed (can only be used once)",
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
                None, // No destructor - PyTorch handles it
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
        if self.ptr.is_null() {
            return Err(PyRuntimeError::new_err("Invalid DLPack tensor pointer"));
        }

        unsafe {
            let tensor = &(*self.ptr).dl_tensor;
            // device_type is an enum, convert to integer
            // kDLCUDA = 2, kDLCPU = 1
            // Ref: https://github.com/dmlc/dlpack/blob/6ea9b3eb64c881f614cd4537f95f0e125a35555c/include/dlpack/dlpack.h#L76-L80
            let device_type = match tensor.device.device_type {
                qdp_core::dlpack::DLDeviceType::kDLCUDA => 2,
                qdp_core::dlpack::DLDeviceType::kDLCPU => 1,
            };
            // Read device_id from DLPack tensor metadata
            Ok((device_type, tensor.device.device_id))
        }
    }
}

impl Drop for QuantumTensor {
    fn drop(&mut self) {
        // Only free if not consumed by __dlpack__
        // If consumed, PyTorch/consumer will call the deleter
        if !self.consumed && !self.ptr.is_null() {
            unsafe {
                // Defensive check: qdp-core always provides a deleter
                debug_assert!(
                    (*self.ptr).deleter.is_some(),
                    "DLManagedTensor from qdp-core should always have a deleter"
                );

                // Call the DLPack deleter to free memory
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

/// Helper to detect PyTorch tensor
fn is_pytorch_tensor(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let type_obj = obj.get_type();
    let name = type_obj.name()?;
    if name != "Tensor" {
        return Ok(false);
    }
    let module = type_obj.module()?;
    let module_name = module.to_str()?;
    Ok(module_name == "torch")
}

/// Helper to validate CPU tensor
fn validate_tensor(tensor: &Bound<'_, PyAny>) -> PyResult<()> {
    if !is_pytorch_tensor(tensor)? {
        return Err(PyRuntimeError::new_err("Object is not a PyTorch Tensor"));
    }

    let device = tensor.getattr("device")?;
    let device_type: String = device.getattr("type")?.extract()?;

    if device_type != "cpu" {
        return Err(PyRuntimeError::new_err(format!(
            "Only CPU tensors are currently supported for this path. Got device: {}",
            device_type
        )));
    }

    Ok(())
}

/// Check if a PyTorch tensor is on a CUDA device
fn is_cuda_tensor(tensor: &Bound<'_, PyAny>) -> PyResult<bool> {
    let device = tensor.getattr("device")?;
    let device_type: String = device.getattr("type")?.extract()?;
    Ok(device_type == "cuda")
}

/// Get the CUDA device index from a PyTorch tensor
fn get_tensor_device_id(tensor: &Bound<'_, PyAny>) -> PyResult<i32> {
    let device = tensor.getattr("device")?;
    let device_index: i32 = device.getattr("index")?.extract()?;
    Ok(device_index)
}

/// Validate a CUDA tensor for direct GPU encoding
/// Checks: dtype=float64, contiguous, non-empty, device_id matches engine
fn validate_cuda_tensor_for_encoding(
    tensor: &Bound<'_, PyAny>,
    expected_device_id: usize,
    encoding_method: &str,
) -> PyResult<()> {
    // Check encoding method support (currently only amplitude is supported for CUDA tensors)
    if encoding_method != "amplitude" {
        return Err(PyRuntimeError::new_err(format!(
            "CUDA tensor encoding currently only supports 'amplitude' method, got '{}'. \
             Use tensor.cpu() to convert to CPU tensor for other encoding methods.",
            encoding_method
        )));
    }

    // Check dtype is float64
    let dtype = tensor.getattr("dtype")?;
    let dtype_str: String = dtype.str()?.extract()?;
    if !dtype_str.contains("float64") {
        return Err(PyRuntimeError::new_err(format!(
            "CUDA tensor must have dtype float64, got {}. Use tensor.to(torch.float64)",
            dtype_str
        )));
    }

    // Check contiguous
    let is_contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
    if !is_contiguous {
        return Err(PyRuntimeError::new_err(
            "CUDA tensor must be contiguous. Use tensor.contiguous()",
        ));
    }

    // Check non-empty
    let numel: usize = tensor.call_method0("numel")?.extract()?;
    if numel == 0 {
        return Err(PyRuntimeError::new_err("CUDA tensor cannot be empty"));
    }

    // Check device matches engine
    let tensor_device_id = get_tensor_device_id(tensor)?;
    if tensor_device_id as usize != expected_device_id {
        return Err(PyRuntimeError::new_err(format!(
            "Device mismatch: tensor is on cuda:{}, but engine is on cuda:{}. \
             Move tensor with tensor.to('cuda:{}')",
            tensor_device_id, expected_device_id, expected_device_id
        )));
    }

    Ok(())
}

/// CUDA tensor information extracted directly from PyTorch tensor
struct CudaTensorInfo {
    data_ptr: *const f64,
    shape: Vec<i64>,
}

/// Extract GPU pointer directly from PyTorch CUDA tensor
///
/// Uses PyTorch's `data_ptr()` and `shape` APIs directly instead of DLPack protocol.
/// This avoids the DLPack capsule lifecycle complexity and potential memory leaks
/// from the capsule renaming pattern.
///
/// # Safety
/// The returned `data_ptr` points to GPU memory owned by the source tensor.
/// The caller must ensure the source tensor remains alive and unmodified
/// for the entire duration that `data_ptr` is in use. Python's GIL ensures
/// the tensor won't be garbage collected during `encode()`, but the caller
/// must not deallocate or resize the tensor while encoding is in progress.
fn extract_cuda_tensor_info(tensor: &Bound<'_, PyAny>) -> PyResult<CudaTensorInfo> {
    // Get GPU pointer directly via tensor.data_ptr()
    let data_ptr_int: isize = tensor.call_method0("data_ptr")?.extract()?;
    if data_ptr_int == 0 {
        return Err(PyRuntimeError::new_err("CUDA tensor has null data pointer"));
    }
    let data_ptr = data_ptr_int as *const f64;

    // Get shape directly via tensor.shape
    let shape_obj = tensor.getattr("shape")?;
    let shape: Vec<i64> = shape_obj.extract()?;

    Ok(CudaTensorInfo { data_ptr, shape })
}

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
    ///     precision: Output precision ("float32" default, or "float64")
    ///
    /// Returns:
    ///     QdpEngine instance
    ///
    /// Raises:
    ///     RuntimeError: If CUDA device initialization fails
    #[new]
    #[pyo3(signature = (device_id=0, precision="float32"))]
    fn new(device_id: usize, precision: &str) -> PyResult<Self> {
        let precision = match precision.to_ascii_lowercase().as_str() {
            "float32" | "f32" | "float" => Precision::Float32,
            "float64" | "f64" | "double" => Precision::Float64,
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unsupported precision '{}'. Use 'float32' (default) or 'float64'.",
                    other
                )));
            }
        };

        let engine = CoreEngine::new_with_precision(device_id, precision)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;
        Ok(Self { engine })
    }

    /// Encode classical data into quantum state (auto-detects input type)
    ///
    /// Args:
    ///     data: Input data - supports:
    ///         - Python list: [1.0, 2.0, 3.0, 4.0]
    ///         - NumPy array: 1D (single sample) or 2D (batch) array
    ///         - PyTorch tensor: CPU tensor (float64 recommended; will be copied to GPU)
    ///         - String path: .parquet, .arrow, .feather, .npy, .pt, .pth, .pb file
    ///         - pathlib.Path: Path object (converted via os.fspath())
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy ("amplitude" default, "angle", or "basis")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack-compatible tensor for zero-copy PyTorch integration
    ///         Shape: [batch_size, 2^num_qubits]
    ///
    /// Example:
    ///     >>> engine = QdpEngine(0)
    ///     >>> # From list
    ///     >>> tensor = engine.encode([1.0, 2.0, 3.0, 4.0], 2)
    ///     >>> # From NumPy batch
    ///     >>> tensor = engine.encode(np.random.randn(64, 4), 2)
    ///     >>> # From file path string
    ///     >>> tensor = engine.encode("data.parquet", 10)
    ///     >>> # From pathlib.Path
    ///     >>> from pathlib import Path
    ///     >>> tensor = engine.encode(Path("data.npy"), 10)
    #[pyo3(signature = (data, num_qubits, encoding_method="amplitude"))]
    fn encode(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        // Check if it's a string path
        if let Ok(path) = data.extract::<String>() {
            return self.encode_from_file(&path, num_qubits, encoding_method);
        }

        // Check if it's a pathlib.Path or os.PathLike object (has __fspath__ method)
        if data.hasattr("__fspath__")? {
            let path: String = data.call_method0("__fspath__")?.extract()?;
            return self.encode_from_file(&path, num_qubits, encoding_method);
        }

        // Check if it's a NumPy array
        if data.hasattr("__array_interface__")? {
            // Get the array's ndim for shape validation
            let ndim: usize = data.getattr("ndim")?.extract()?;

            match ndim {
                1 => {
                    // 1D array: single sample encoding (zero-copy if already contiguous)
                    let array_1d = data.extract::<PyReadonlyArray1<f64>>().map_err(|_| {
                        PyRuntimeError::new_err(
                            "Failed to extract 1D NumPy array. Ensure dtype is float64.",
                        )
                    })?;
                    let data_slice = array_1d.as_slice().map_err(|_| {
                        PyRuntimeError::new_err("NumPy array must be contiguous (C-order)")
                    })?;
                    let ptr = self
                        .engine
                        .encode(data_slice, num_qubits, encoding_method)
                        .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                    return Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    });
                }
                2 => {
                    // 2D array: batch encoding (zero-copy if already contiguous)
                    let array_2d = data.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
                        PyRuntimeError::new_err(
                            "Failed to extract 2D NumPy array. Ensure dtype is float64.",
                        )
                    })?;
                    let shape = array_2d.shape();
                    let num_samples = shape[0];
                    let sample_size = shape[1];
                    let data_slice = array_2d.as_slice().map_err(|_| {
                        PyRuntimeError::new_err("NumPy array must be contiguous (C-order)")
                    })?;
                    let ptr = self
                        .engine
                        .encode_batch(
                            data_slice,
                            num_samples,
                            sample_size,
                            num_qubits,
                            encoding_method,
                        )
                        .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                    return Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    });
                }
                _ => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Unsupported array shape: {}D. Expected 1D array for single sample \
                         encoding or 2D array (batch_size, features) for batch encoding.",
                        ndim
                    )));
                }
            }
        }

        // Check if it's a PyTorch tensor
        if is_pytorch_tensor(data)? {
            // Check if it's a CUDA tensor - use zero-copy GPU encoding
            if is_cuda_tensor(data)? {
                // Validate CUDA tensor for direct GPU encoding
                validate_cuda_tensor_for_encoding(
                    data,
                    self.engine.device().ordinal(),
                    encoding_method,
                )?;

                // Extract GPU pointer directly from PyTorch tensor
                let tensor_info = extract_cuda_tensor_info(data)?;

                let ndim: usize = data.call_method0("dim")?.extract()?;

                match ndim {
                    1 => {
                        // 1D CUDA tensor: single sample encoding
                        let input_len = tensor_info.shape[0] as usize;
                        // SAFETY: tensor_info.data_ptr was obtained via PyTorch's data_ptr() from a
                        // valid CUDA tensor. The tensor remains alive during this call
                        // (held by Python's GIL), and we validated dtype/contiguity/device above.
                        let ptr = unsafe {
                            self.engine
                                .encode_from_gpu_ptr(
                                    tensor_info.data_ptr,
                                    input_len,
                                    num_qubits,
                                    encoding_method,
                                )
                                .map_err(|e| {
                                    PyRuntimeError::new_err(format!("Encoding failed: {}", e))
                                })?
                        };
                        return Ok(QuantumTensor {
                            ptr,
                            consumed: false,
                        });
                    }
                    2 => {
                        // 2D CUDA tensor: batch encoding
                        let num_samples = tensor_info.shape[0] as usize;
                        let sample_size = tensor_info.shape[1] as usize;
                        // SAFETY: Same as above - pointer from validated PyTorch CUDA tensor
                        let ptr = unsafe {
                            self.engine
                                .encode_batch_from_gpu_ptr(
                                    tensor_info.data_ptr,
                                    num_samples,
                                    sample_size,
                                    num_qubits,
                                    encoding_method,
                                )
                                .map_err(|e| {
                                    PyRuntimeError::new_err(format!("Encoding failed: {}", e))
                                })?
                        };
                        return Ok(QuantumTensor {
                            ptr,
                            consumed: false,
                        });
                    }
                    _ => {
                        return Err(PyRuntimeError::new_err(format!(
                            "Unsupported CUDA tensor shape: {}D. Expected 1D tensor for single \
                             sample encoding or 2D tensor (batch_size, features) for batch encoding.",
                            ndim
                        )));
                    }
                }
            }

            // CPU tensor path (existing code)
            validate_tensor(data)?;
            // PERF: Avoid Tensor -> Python list -> Vec deep copies.
            //
            // For CPU tensors, `tensor.detach().numpy()` returns a NumPy view that shares the same
            // underlying memory (zero-copy) when the tensor is C-contiguous. We can then borrow a
            // `&[f64]` directly via pyo3-numpy.
            let ndim: usize = data.call_method0("dim")?.extract()?;
            let numpy_view = data
                .call_method0("detach")?
                .call_method0("numpy")
                .map_err(|_| {
                    PyRuntimeError::new_err(
                        "Failed to convert torch.Tensor to NumPy view. Ensure the tensor is on CPU \
                         and does not require grad (try: tensor = tensor.detach().cpu())",
                    )
                })?;

            let array = numpy_view
                .extract::<PyReadonlyArrayDyn<f64>>()
                .map_err(|_| {
                    PyRuntimeError::new_err(
                        "Failed to extract NumPy view as float64 array. Ensure dtype is float64 \
                         (try: tensor = tensor.to(torch.float64))",
                    )
                })?;

            let data_slice = array.as_slice().map_err(|_| {
                PyRuntimeError::new_err(
                    "Tensor must be contiguous (C-order) to get zero-copy slice \
                     (try: tensor = tensor.contiguous())",
                )
            })?;

            match ndim {
                1 => {
                    // 1D tensor: single sample encoding
                    let ptr = self
                        .engine
                        .encode(data_slice, num_qubits, encoding_method)
                        .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                    return Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    });
                }
                2 => {
                    // 2D tensor: batch encoding
                    let shape = array.shape();
                    if shape.len() != 2 {
                        return Err(PyRuntimeError::new_err(format!(
                            "Unsupported tensor shape: {}D. Expected 2D tensor (batch_size, features).",
                            shape.len()
                        )));
                    }
                    let num_samples = shape[0];
                    let sample_size = shape[1];
                    let ptr = self
                        .engine
                        .encode_batch(
                            data_slice,
                            num_samples,
                            sample_size,
                            num_qubits,
                            encoding_method,
                        )
                        .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                    return Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    });
                }
                _ => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Unsupported tensor shape: {}D. Expected 1D tensor for single sample \
                         encoding or 2D tensor (batch_size, features) for batch encoding.",
                        ndim
                    )));
                }
            }
        }

        // Fallback: try to extract as Vec<f64> (Python list)
        if let Ok(vec_data) = data.extract::<Vec<f64>>() {
            let ptr = self
                .engine
                .encode(&vec_data, num_qubits, encoding_method)
                .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
            return Ok(QuantumTensor {
                ptr,
                consumed: false,
            });
        }

        Err(PyRuntimeError::new_err(
            "Unsupported data type. Expected: list, NumPy array, PyTorch tensor, or file path",
        ))
    }

    /// Internal helper to encode from file based on extension
    fn encode_from_file(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ptr = if path.ends_with(".parquet") {
            self.engine
                .encode_from_parquet(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from parquet failed: {}", e))
                })?
        } else if path.ends_with(".arrow") || path.ends_with(".feather") {
            self.engine
                .encode_from_arrow_ipc(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from Arrow IPC failed: {}", e))
                })?
        } else if path.ends_with(".npy") {
            self.engine
                .encode_from_numpy(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from NumPy failed: {}", e))
                })?
        } else if path.ends_with(".pt") || path.ends_with(".pth") {
            self.engine
                .encode_from_torch(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from PyTorch failed: {}", e))
                })?
        } else if path.ends_with(".pb") {
            self.engine
                .encode_from_tensorflow(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from TensorFlow failed: {}", e))
                })?
        } else {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported file format. Expected .parquet, .arrow, .feather, .npy, .pt, .pth, or .pb, got: {}",
                path
            )));
        };

        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    /// Encode from TensorFlow TensorProto file
    ///
    /// Args:
    ///     path: Path to TensorProto file (.pb)
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy (currently only "amplitude")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack tensor containing all encoded states
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> batched = engine.encode_from_tensorflow("data.pb", 16, "amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(batched)  # Shape: [200, 65536]
    fn encode_from_tensorflow(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ptr = self
            .engine
            .encode_from_tensorflow(path, num_qubits, encoding_method)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Encoding from TensorFlow failed: {}", e))
            })?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }
}

/// Quantum Data Plane (QDP) Python module
///
/// GPU-accelerated quantum data encoding with DLPack integration.
#[pymodule]
fn _qdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QdpEngine>()?;
    m.add_class::<QuantumTensor>()?;
    Ok(())
}
