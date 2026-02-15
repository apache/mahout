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

// Loader bindings (Linux only; qdp-core pipeline types only built on Linux)
#[cfg(target_os = "linux")]
mod loader_impl {
    use crate::tensor::QuantumTensor;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;
    use qdp_core::{PipelineConfig, PipelineIterator, QdpEngine as CoreEngine};

    /// Rust-backed iterator yielding one QuantumTensor per batch; used by QuantumDataLoader.
    #[pyclass]
    pub struct PyQuantumLoader {
        pub inner: Option<PipelineIterator>,
    }

    impl PyQuantumLoader {
        pub fn new(inner: Option<PipelineIterator>) -> Self {
            Self { inner }
        }
    }

    #[pymethods]
    impl PyQuantumLoader {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<QuantumTensor> {
            let mut iter: PipelineIterator = match slf.inner.take() {
                Some(i) => i,
                None => return Err(pyo3::exceptions::PyStopIteration::new_err("")),
            };
            // Call next_batch without releasing GIL (return type *mut DLManagedTensor is !Send).
            let result = iter.next_batch();
            match result {
                Ok(Some(ptr)) => {
                    slf.inner = Some(iter);
                    Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    })
                }
                Ok(None) => {
                    // Exhausted; do not put iterator back
                    Err(pyo3::exceptions::PyStopIteration::new_err(""))
                }
                Err(e) => {
                    slf.inner = Some(iter);
                    Err(PyRuntimeError::new_err(format!(
                        "Pipeline next_batch failed: {}",
                        e
                    )))
                }
            }
        }
    }

    /// Build PipelineConfig from Python args. device_id is 0 (engine does not expose it); iterator uses engine clone with correct device.
    pub fn config_from_args(
        _engine: &CoreEngine,
        batch_size: usize,
        num_qubits: u32,
        encoding_method: &str,
        total_batches: usize,
        seed: Option<u64>,
    ) -> PipelineConfig {
        PipelineConfig {
            device_id: 0,
            num_qubits,
            batch_size,
            total_batches,
            encoding_method: encoding_method.to_string(),
            seed,
            warmup_batches: 0,
        }
    }

    /// Resolve path from Python str or pathlib.Path (__fspath__).
    pub fn path_from_py(path: &Bound<'_, PyAny>) -> PyResult<String> {
        path.extract::<String>().or_else(|_| {
            path.call_method0("__fspath__")
                .and_then(|m| m.extract::<String>())
        })
    }
}

#[cfg(target_os = "linux")]
pub use loader_impl::{PyQuantumLoader, config_from_args, path_from_py};
