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

use crate::input::PyInput;
use crate::tensor::QuantumTensor;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use qdp_core::{Dtype, Encoding, QdpEngine as CoreEngine};

#[cfg(target_os = "linux")]
use crate::loader::{
    PyQuantumLoader, config_from_args, parse_dtype, parse_null_handling, path_from_py,
};

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
        let precision =
            Dtype::from_str_ci(precision).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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
    ///           or CUDA tensor for zero-copy encoding
    ///         - String path: .parquet, .arrow, .feather, .npy, .pt, .pth, .pb file
    ///         - pathlib.Path: Path object (converted via os.fspath())
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy ("amplitude" default, "angle", or "basis")
    ///         CUDA tensor note:
    ///         - amplitude accepts float64 and float32
    ///         - angle accepts float64 generally, plus float32 for 1D single-sample tensors
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
    /// Encode `data` with `encoding_method` into a `[samples, 2**num_qubits]`
    /// state tensor on the GPU.
    ///
    /// `data` may be a list, a NumPy array, a CPU or CUDA PyTorch tensor, or a
    /// path to a `.parquet`, `.arrow`, `.feather`, `.npy`, `.pt`, `.pth` or
    /// `.pb` file. CUDA tensors are read in place on their current stream.
    #[pyo3(signature = (data, num_qubits, encoding_method="amplitude"))]
    fn encode(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        if let Ok(path) = data.extract::<String>() {
            return self.encode_from_file(&path, num_qubits, encoding_method);
        }
        if data.hasattr("__fspath__")? {
            let path: String = data.call_method0("__fspath__")?.extract()?;
            return self.encode_from_file(&path, num_qubits, encoding_method);
        }

        let encoding = Encoding::from_str_ci(encoding_method)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let input = PyInput::from_py(data, encoding, self.engine.device().ordinal())?;
        let (input, shape) = input.as_input()?;
        let ptr = self
            .engine
            .encode(input, shape, num_qubits, encoding)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    fn encode_from_file(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        #[cfg(feature = "remote-io")]
        let _resolved;
        #[cfg(feature = "remote-io")]
        let path = {
            _resolved = qdp_core::remote::resolve_path(path).map_err(|e| {
                PyRuntimeError::new_err(format!("Remote path resolution failed: {}", e))
            })?;
            _resolved
                .path
                .to_str()
                .ok_or_else(|| PyRuntimeError::new_err("Resolved path is not valid UTF-8"))?
        };

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
        // Synthetic data is generated in-process for throughput benchmarking, so it
        // defaults to f32 (PipelineConfig::normalize downgrades to f64 for encodings
        // without an f32 batch path). This is deliberate and unrelated to the file
        // loaders, which default to f64 to keep user-supplied data lossless.
        let config = config_from_args(
            &self.engine,
            batch_size,
            num_qubits,
            encoding_method,
            total_batches,
            seed,
            nh,
            Dtype::Float32,
        )?;
        let iter = qdp_core::PipelineIterator::new_synthetic(self.engine.clone(), config).map_err(
            |e| PyRuntimeError::new_err(format!("create_synthetic_loader failed: {}", e)),
        )?;
        Ok(PyQuantumLoader::new(Some(iter)))
    }

    #[cfg(target_os = "linux")]
    /// Create a file-backed pipeline iterator (full read then batch; for QuantumDataLoader.source_file(path)).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, batch_size, num_qubits, encoding_method, batch_limit=None, null_handling=None, dtype=None))]
    fn create_file_loader(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        batch_size: usize,
        num_qubits: u32,
        encoding_method: &str,
        batch_limit: Option<usize>,
        null_handling: Option<&str>,
        dtype: Option<&str>,
    ) -> PyResult<PyQuantumLoader> {
        let path_str = path_from_py(path)?;
        let batch_limit = batch_limit.unwrap_or(usize::MAX);
        let nh = parse_null_handling(null_handling)?;
        let dt = parse_dtype(dtype)?;
        let config = config_from_args(
            &self.engine,
            batch_size,
            num_qubits,
            encoding_method,
            0,
            None,
            nh,
            dt,
        )?;
        let engine = self.engine.clone();
        // Resolve remote URLs before detaching from GIL. The _resolved guard keeps the
        // temp file alive until after the file is fully read inside py.detach.
        #[cfg(feature = "remote-io")]
        let _resolved = qdp_core::remote::resolve_path(path_str.as_str()).map_err(|e| {
            PyRuntimeError::new_err(format!("Remote path resolution failed: {}", e))
        })?;
        #[cfg(feature = "remote-io")]
        let path_str = _resolved.path.to_string_lossy().into_owned();
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
    #[pyo3(signature = (path, batch_size, num_qubits, encoding_method, batch_limit=None, null_handling=None, dtype=None))]
    fn create_streaming_file_loader(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        batch_size: usize,
        num_qubits: u32,
        encoding_method: &str,
        batch_limit: Option<usize>,
        null_handling: Option<&str>,
        dtype: Option<&str>,
    ) -> PyResult<PyQuantumLoader> {
        let path_str = path_from_py(path)?;
        let batch_limit = batch_limit.unwrap_or(usize::MAX);
        let nh = parse_null_handling(null_handling)?;
        let dt = parse_dtype(dtype)?;
        let config = config_from_args(
            &self.engine,
            batch_size,
            num_qubits,
            encoding_method,
            0,
            None,
            nh,
            dt,
        )?;
        let engine = self.engine.clone();
        // Resolve remote URLs before detaching from GIL. The _resolved guard keeps the
        // temp file alive; the streaming reader's open fd preserves data after drop.
        #[cfg(feature = "remote-io")]
        let _resolved = qdp_core::remote::resolve_path(path_str.as_str()).map_err(|e| {
            PyRuntimeError::new_err(format!("Remote path resolution failed: {}", e))
        })?;
        #[cfg(feature = "remote-io")]
        let path_str = _resolved.path.to_string_lossy().into_owned();
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
