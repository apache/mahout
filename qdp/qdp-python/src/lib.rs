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

mod constants;
mod dlpack;
mod engine;
mod loader;
mod pytorch;
mod tensor;

use engine::QdpEngine;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tensor::QuantumTensor;

#[cfg(target_os = "linux")]
use loader::PyQuantumLoader;

#[cfg(target_os = "linux")]
#[pyfunction]
#[pyo3(signature = (device_id, num_qubits, batch_size, total_batches, encoding_method, warmup_batches=0, seed=None, dtype="f64"))]
#[allow(clippy::too_many_arguments)]
fn run_throughput_pipeline_py(
    py: Python<'_>,
    device_id: usize,
    num_qubits: u32,
    batch_size: usize,
    total_batches: usize,
    encoding_method: String,
    warmup_batches: usize,
    seed: Option<u64>,
    dtype: &str,
) -> PyResult<(f64, f64, f64)> {
    let config = qdp_core::PipelineConfig {
        device_id,
        num_qubits,
        batch_size,
        total_batches,
        encoding: qdp_core::Encoding::from_str_ci(&encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid encoding_method: {e}")))?,
        seed,
        warmup_batches,
        null_handling: qdp_core::NullHandling::default(),
        dtype: qdp_core::Dtype::from_str_ci(dtype)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid dtype: {e}")))?,
        prefetch_depth: 16,
    };
    let result = py
        .detach(|| qdp_core::run_throughput_pipeline(&config))
        .map_err(|e| PyRuntimeError::new_err(format!("Pipeline failed: {e}")))?;
    Ok((
        result.duration_sec,
        result.vectors_per_sec,
        result.latency_ms_per_vector,
    ))
}

/// Estimate the host and device memory a pipeline configuration would need.
///
/// Returns ``(cpu_prefetch_bytes, gpu_state_bytes, total_bytes)``. Pure config
/// arithmetic: allocates nothing, touches no device, and is therefore callable on a
/// stub build or a host with no GPU -- which is the point, since the intended use is
/// sizing a configuration before building a loader that would reject it.
///
/// ``gpu_state_bytes`` is the figure the loader's memory guard compares against free
/// VRAM when iteration starts, with one caveat worth stating here: the guard budgets
/// the *wider* of this ``dtype`` and the engine's precision (and always f64 for
/// ``basis`` read from a file, whose inputs are integer state indices read as f64;
/// synthetic basis data is budgeted at the requested ``dtype``). Pass the
/// engine's precision as ``dtype`` when the two differ, or the estimate will be half
/// what the guard uses.
///
/// Every failure the estimator itself reports is a bad argument -- an unknown encoding
/// or dtype name, a ``num_qubits`` whose 2^n state vector is not representable, or a
/// product that overflows -- and raises ``ValueError``. Arguments that cannot be
/// converted at the boundary fail earlier and differently: a negative ``num_qubits`` or
/// ``batch_size`` raises ``OverflowError``, a non-integer raises ``TypeError``.
#[pyfunction]
#[pyo3(signature = (num_qubits, batch_size, encoding_method="amplitude", dtype="f64", prefetch_depth=16))]
fn estimate_memory(
    num_qubits: u32,
    batch_size: usize,
    encoding_method: &str,
    dtype: &str,
    prefetch_depth: usize,
) -> PyResult<(u64, u64, u64)> {
    let encoding = qdp_core::Encoding::from_str_ci(encoding_method)
        .map_err(|e| PyValueError::new_err(format!("Invalid encoding_method: {e}")))?;
    let dtype = qdp_core::Dtype::from_str_ci(dtype)
        .map_err(|e| PyValueError::new_err(format!("Invalid dtype: {e}")))?;
    let estimate =
        qdp_core::estimate_memory(encoding, num_qubits, batch_size, dtype, prefetch_depth)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        estimate.cpu_prefetch_bytes,
        estimate.gpu_state_bytes,
        estimate.total(),
    ))
}

/// Returns ``True`` if a usable CUDA device is available to the native engine.
///
/// This reflects whether GPU work can actually run -- it is ``False`` for a
/// stub build (the extension built without the CUDA toolkit) or a host with no
/// CUDA device, and ``True`` only when the native runtime reports a device.
/// Importability of this module only means it was built, which a stub build
/// makes possible without a GPU.
#[pyfunction]
fn cuda_available() -> bool {
    qdp_core::cuda_runtime_available()
}

/// Quantum Data Plane (QDP) Python module
///
/// GPU-accelerated quantum data encoding with DLPack integration.
#[pymodule]
fn _qdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Respect RUST_LOG for Rust log output; try_init() is no-op if already initialized.
    let _ = env_logger::Builder::from_default_env().try_init();

    m.add_class::<QdpEngine>()?;
    m.add_class::<QuantumTensor>()?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_memory, m)?)?;
    #[cfg(target_os = "linux")]
    m.add_class::<PyQuantumLoader>()?;
    #[cfg(target_os = "linux")]
    m.add_function(wrap_pyfunction!(run_throughput_pipeline_py, m)?)?;
    Ok(())
}
