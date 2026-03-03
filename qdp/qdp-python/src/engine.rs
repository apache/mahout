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

use crate::dlpack::extract_dlpack_tensor;
use crate::pytorch::{
    extract_cuda_tensor_info, get_tensor_device_id, get_torch_cuda_stream_ptr, is_cuda_tensor,
    is_pytorch_tensor, validate_cuda_tensor_for_encoding, validate_shape, validate_tensor,
};
use crate::tensor::QuantumTensor;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use qdp_core::{Precision, QdpEngine as CoreEngine};

#[cfg(target_os = "linux")]
use crate::loader::{PyQuantumLoader, config_from_args, parse_null_handling, path_from_py};

/// PyO3 wrapper for QdpEngine
///
/// Provides Python bindings for GPU-accelerated quantum state encoding.
#[pyclass]
pub struct QdpEngine {
    pub engine: CoreEngine,
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
            return self.encode_from_numpy(data, num_qubits, encoding_method);
        }

        // Check if it's a PyTorch tensor
        if is_pytorch_tensor(data)? {
            // Check if it's a CUDA tensor - use zero-copy GPU encoding
            if is_cuda_tensor(data)? {
                return self._encode_from_cuda_tensor(data, num_qubits, encoding_method);
            }
            // CPU PyTorch tensor path
            return self.encode_from_pytorch(data, num_qubits, encoding_method);
        }

        // Fallback: try to extract as Vec<f64> (Python list)
        self.encode_from_list(data, num_qubits, encoding_method)
    }

    /// Encode from NumPy array (1D or 2D)
    fn encode_from_numpy(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ndim: usize = data.getattr("ndim")?.extract()?;
        validate_shape(ndim, "array")?;

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
                Ok(QuantumTensor {
                    ptr,
                    consumed: false,
                })
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
                Ok(QuantumTensor {
                    ptr,
                    consumed: false,
                })
            }
            _ => unreachable!("validate_shape() should have caught invalid ndim"),
        }
    }

    /// Encode from PyTorch tensor (1D or 2D)
    fn encode_from_pytorch(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        // Check if it's a CUDA tensor - use zero-copy GPU encoding via DLPack
        if is_cuda_tensor(data)? {
            // Validate CUDA tensor for direct GPU encoding
            validate_cuda_tensor_for_encoding(
                data,
                self.engine.device().ordinal(),
                encoding_method,
            )?;

            // Extract GPU pointer via DLPack (RAII wrapper ensures deleter is called)
            let dlpack_info = extract_dlpack_tensor(data.py(), data)?;

            // ensure PyTorch API and DLPack metadata agree on device ID
            let pytorch_device_id = get_tensor_device_id(data)?;
            if dlpack_info.device_id != pytorch_device_id {
                return Err(PyRuntimeError::new_err(format!(
                    "Device ID mismatch: PyTorch reports device {}, but DLPack metadata reports {}. \
                     This indicates an inconsistency between PyTorch and DLPack device information.",
                    pytorch_device_id, dlpack_info.device_id
                )));
            }

            let ndim: usize = data.call_method0("dim")?.extract()?;
            validate_shape(ndim, "CUDA tensor")?;

            match ndim {
                1 => {
                    // 1D CUDA tensor: single sample encoding
                    let input_len = dlpack_info.shape[0] as usize;
                    // SAFETY: dlpack_info.data_ptr was validated via DLPack protocol from a
                    // valid PyTorch CUDA tensor. The tensor remains alive during this call
                    // (held by Python's GIL), and we validated dtype/contiguity/device above.
                    // The DLPackTensorInfo RAII wrapper will call deleter when dropped.
                    let ptr = unsafe {
                        self.engine
                            .encode_from_gpu_ptr(
                                dlpack_info.data_ptr,
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
                    let num_samples = dlpack_info.shape[0] as usize;
                    let sample_size = dlpack_info.shape[1] as usize;
                    // SAFETY: Same as above - pointer from validated DLPack tensor
                    let ptr = unsafe {
                        self.engine
                            .encode_batch_from_gpu_ptr(
                                dlpack_info.data_ptr,
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
                _ => unreachable!("validate_shape() should have caught invalid ndim"),
            }
        }

        // CPU tensor path
        validate_tensor(data)?;
        // PERF: Avoid Tensor -> Python list -> Vec deep copies.
        //
        // For CPU tensors, `tensor.detach().numpy()` returns a NumPy view that shares the same
        // underlying memory (zero-copy) when the tensor is C-contiguous. We can then borrow a
        // `&[f64]` directly via pyo3-numpy.
        let ndim: usize = data.call_method0("dim")?.extract()?;
        validate_shape(ndim, "tensor")?;
        let numpy_view = data
            .call_method0("detach")?
            .call_method0("numpy")
            .map_err(|_| {
                PyRuntimeError::new_err(
                    "Failed to convert torch.Tensor to NumPy view. Ensure the tensor is on CPU \
                     and does not require grad (try: tensor = tensor.detach().cpu())",
                )
            })?;

        match ndim {
            1 => {
                // 1D tensor: single sample encoding
                let array_1d = numpy_view.extract::<PyReadonlyArray1<f64>>().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Failed to extract NumPy view as float64 array. Ensure dtype is float64 \
                             (try: tensor = tensor.to(torch.float64))",
                    )
                })?;
                let data_slice = array_1d.as_slice().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Tensor must be contiguous (C-order) to get zero-copy slice \
                         (try: tensor = tensor.contiguous())",
                    )
                })?;
                let ptr = self
                    .engine
                    .encode(data_slice, num_qubits, encoding_method)
                    .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                Ok(QuantumTensor {
                    ptr,
                    consumed: false,
                })
            }
            2 => {
                // 2D tensor: batch encoding
                let array_2d = numpy_view.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Failed to extract NumPy view as float64 array. Ensure dtype is float64 \
                             (try: tensor = tensor.to(torch.float64))",
                    )
                })?;
                let shape = array_2d.shape();
                let num_samples = shape[0];
                let sample_size = shape[1];
                let data_slice = array_2d.as_slice().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Tensor must be contiguous (C-order) to get zero-copy slice \
                         (try: tensor = tensor.contiguous())",
                    )
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
                Ok(QuantumTensor {
                    ptr,
                    consumed: false,
                })
            }
            _ => unreachable!("validate_shape() should have caught invalid ndim"),
        }
    }

    /// Encode from Python list
    fn encode_from_list(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let vec_data = data.extract::<Vec<f64>>().map_err(|_| {
            PyRuntimeError::new_err(
                "Unsupported data type. Expected: list, NumPy array, PyTorch tensor, or file path",
            )
        })?;
        let ptr = self
            .engine
            .encode(&vec_data, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
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

    /// Encode directly from a PyTorch CUDA tensor. Internal helper.
    ///
    /// Dispatches to the core f32 GPU pointer API for 1D float32 amplitude encoding,
    /// or to the float64/basis GPU pointer APIs for other dtypes and batch encoding.
    fn _encode_from_cuda_tensor(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        validate_cuda_tensor_for_encoding(data, self.engine.device().ordinal(), encoding_method)?;

        let dtype = data.getattr("dtype")?;
        let dtype_str: String = dtype.str()?.extract()?;
        let dtype_str_lower = dtype_str.to_ascii_lowercase();
        let is_f32 = dtype_str_lower.contains("float32");
        let method = encoding_method.to_ascii_lowercase();
        let ndim: usize = data.call_method0("dim")?.extract()?;

        if method.as_str() == "amplitude" && is_f32 {
            match ndim {
                1 => {
                    let input_len: usize = data.call_method0("numel")?.extract()?;
                    let stream_ptr = get_torch_cuda_stream_ptr(data)?;
                    let data_ptr_u64: u64 = data.call_method0("data_ptr")?.extract()?;
                    let data_ptr = data_ptr_u64 as *const f32;

                    let ptr = unsafe {
                        self.engine
                            .encode_from_gpu_ptr_f32_with_stream(
                                data_ptr, input_len, num_qubits, stream_ptr,
                            )
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!(
                                    "Encoding failed (float32 amplitude): {}",
                                    e
                                ))
                            })?
                    };

                    Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    })
                }
                2 => Err(PyRuntimeError::new_err(
                    "CUDA float32 batch amplitude encoding is not yet supported. \
                     Use float64 (tensor.to(torch.float64)) or encode samples individually.",
                )),
                _ => Err(PyRuntimeError::new_err(format!(
                    "Unsupported CUDA tensor shape: {}D. Expected 1D tensor for single \
                     sample encoding or 2D tensor (batch_size, features) for batch encoding.",
                    ndim
                ))),
            }
        } else {
            let tensor_info = extract_cuda_tensor_info(data)?;
            let stream_ptr = get_torch_cuda_stream_ptr(data)?;

            match ndim {
                1 => {
                    let input_len = tensor_info.shape[0] as usize;
                    let ptr = unsafe {
                        self.engine
                            .encode_from_gpu_ptr_with_stream(
                                tensor_info.data_ptr as *const std::ffi::c_void,
                                input_len,
                                num_qubits,
                                encoding_method,
                                stream_ptr,
                            )
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("Encoding failed: {}", e))
                            })?
                    };
                    Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    })
                }
                2 => {
                    let num_samples = tensor_info.shape[0] as usize;
                    let sample_size = tensor_info.shape[1] as usize;
                    let ptr = unsafe {
                        self.engine
                            .encode_batch_from_gpu_ptr_with_stream(
                                tensor_info.data_ptr as *const std::ffi::c_void,
                                num_samples,
                                sample_size,
                                num_qubits,
                                encoding_method,
                                stream_ptr,
                            )
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("Encoding failed: {}", e))
                            })?
                    };
                    Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    })
                }
                _ => Err(PyRuntimeError::new_err(format!(
                    "Unsupported CUDA tensor shape: {}D. Expected 1D tensor for single \
                     sample encoding or 2D tensor (batch_size, features) for batch encoding.",
                    ndim
                ))),
            }
        }
    }

    // --- Loader factory methods (Linux only) ---
    #[cfg(target_os = "linux")]
    /// Create a synthetic-data pipeline iterator (for QuantumDataLoader.source_synthetic()).
    #[pyo3(signature = (total_batches, batch_size, num_qubits, encoding_method, seed=None, null_handling=None))]
    fn create_synthetic_loader(
        &self,
        total_batches: usize,
        batch_size: usize,
        num_qubits: u32,
        encoding_method: &str,
        seed: Option<u64>,
        null_handling: Option<&str>,
    ) -> PyResult<PyQuantumLoader> {
        let nh = parse_null_handling(null_handling)?;
        let config = config_from_args(
            &self.engine,
            batch_size,
            num_qubits,
            encoding_method,
            total_batches,
            seed,
            nh,
        );
        let iter = qdp_core::PipelineIterator::new_synthetic(self.engine.clone(), config).map_err(
            |e| PyRuntimeError::new_err(format!("create_synthetic_loader failed: {}", e)),
        )?;
        Ok(PyQuantumLoader::new(Some(iter)))
    }

    #[cfg(target_os = "linux")]
    /// Create a file-backed pipeline iterator (full read then batch; for QuantumDataLoader.source_file(path)).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, batch_size, num_qubits, encoding_method, batch_limit=None, null_handling=None))]
    fn create_file_loader(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        batch_size: usize,
        num_qubits: u32,
        encoding_method: &str,
        batch_limit: Option<usize>,
        null_handling: Option<&str>,
    ) -> PyResult<PyQuantumLoader> {
        let path_str = path_from_py(path)?;
        let batch_limit = batch_limit.unwrap_or(usize::MAX);
        let nh = parse_null_handling(null_handling)?;
        let config = config_from_args(
            &self.engine,
            batch_size,
            num_qubits,
            encoding_method,
            0,
            None,
            nh,
        );
        let engine = self.engine.clone();
        let iter = py
            .detach(|| {
                qdp_core::PipelineIterator::new_from_file(
                    engine,
                    path_str.as_str(),
                    config,
                    batch_limit,
                )
            })
            .map_err(|e| PyRuntimeError::new_err(format!("create_file_loader failed: {}", e)))?;
        Ok(PyQuantumLoader::new(Some(iter)))
    }

    #[cfg(target_os = "linux")]
    /// Create a streaming Parquet pipeline iterator (for QuantumDataLoader.source_file(path, streaming=True)).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, batch_size, num_qubits, encoding_method, batch_limit=None, null_handling=None))]
    fn create_streaming_file_loader(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        batch_size: usize,
        num_qubits: u32,
        encoding_method: &str,
        batch_limit: Option<usize>,
        null_handling: Option<&str>,
    ) -> PyResult<PyQuantumLoader> {
        let path_str = path_from_py(path)?;
        let batch_limit = batch_limit.unwrap_or(usize::MAX);
        let nh = parse_null_handling(null_handling)?;
        let config = config_from_args(
            &self.engine,
            batch_size,
            num_qubits,
            encoding_method,
            0,
            None,
            nh,
        );
        let engine = self.engine.clone();
        let iter = py
            .detach(|| {
                qdp_core::PipelineIterator::new_from_file_streaming(
                    engine,
                    path_str.as_str(),
                    config,
                    batch_limit,
                )
            })
            .map_err(|e| {
                PyRuntimeError::new_err(format!("create_streaming_file_loader failed: {}", e))
            })?;
        Ok(PyQuantumLoader::new(Some(iter)))
    }
}
