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

//! Python bindings for `qdp_core::preprocessing::Preprocessor`.
//!
//! These are pure CPU helpers (no CUDA calls), so they work in a stub build
//! of the extension on a machine without a GPU.

use std::borrow::Cow;

use numpy::ndarray::Dimension;
use numpy::{
    AllowTypeChange, IntoPyArray, PyArray1, PyArrayLike1, PyArrayLikeDyn, PyReadonlyArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use qdp_core::MahoutError;
use qdp_core::preprocessing::Preprocessor as CorePreprocessor;

/// Map a core error to the Python exception type used by the rest of the
/// bindings (see `engine.rs`): `RuntimeError` carrying the core message.
fn to_py_err(e: MahoutError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Borrow the array as a row-major `&[f64]`, copying only when needed.
///
/// `PyReadonlyArray::as_slice` also succeeds for Fortran-ordered arrays and
/// then returns the elements in *memory* (column-major) order, which would
/// silently reorder a 2D batch. Only C-contiguous arrays are borrowed; any
/// other layout is copied in logical (row-major) order.
fn row_major<'a, D: Dimension>(arr: &'a PyReadonlyArray<'_, f64, D>) -> Cow<'a, [f64]> {
    match arr.as_slice() {
        Ok(slice) if arr.is_c_contiguous() => Cow::Borrowed(slice),
        _ => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

/// Flatten a 1D (already flat) or 2D `(num_samples, sample_size)` batch.
fn flatten_batch<'a>(
    arr: &'a PyReadonlyArray<'_, f64, numpy::IxDyn>,
    num_samples: usize,
    sample_size: usize,
) -> PyResult<Cow<'a, [f64]>> {
    let shape = arr.shape();
    match shape.len() {
        1 => {}
        2 if shape[0] == num_samples && shape[1] == sample_size => {}
        2 => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid input: batch_data has shape ({}, {}) but num_samples={} \
                 and sample_size={}",
                shape[0], shape[1], num_samples, sample_size
            )));
        }
        n => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid input: batch_data must be 1D (flattened) or 2D \
                 (num_samples, sample_size), got {}D",
                n
            )));
        }
    }
    Ok(row_major(arr))
}

/// CPU-side input validation and L2 norms shared by the QDP encoders.
///
/// Static-method binding of ``qdp_core::preprocessing::Preprocessor``.
/// Pure CPU: works without a GPU, including in a CUDA-stub build.
#[pyclass(name = "Preprocessor", frozen)]
pub struct PyPreprocessor;

#[pymethods]
impl PyPreprocessor {
    /// Check a single sample against ``num_qubits`` (non-empty, at most
    /// ``2**num_qubits`` values, 1 <= num_qubits <= 30).
    ///
    /// Raises RuntimeError if the input is invalid.
    #[staticmethod]
    #[pyo3(signature = (data, num_qubits))]
    fn validate_input(
        data: PyArrayLike1<'_, f64, AllowTypeChange>,
        num_qubits: usize,
    ) -> PyResult<()> {
        let values = row_major(&data);
        CorePreprocessor::validate_input(&values, num_qubits).map_err(to_py_err)
    }

    /// Return the L2 norm of a single sample.
    ///
    /// Raises RuntimeError if the norm is zero, NaN or infinite.
    #[staticmethod]
    #[pyo3(signature = (data,))]
    fn calculate_l2_norm(data: PyArrayLike1<'_, f64, AllowTypeChange>) -> PyResult<f64> {
        let values = row_major(&data);
        CorePreprocessor::calculate_l2_norm(&values).map_err(to_py_err)
    }

    /// Check a batch of ``num_samples`` samples of ``sample_size`` values.
    ///
    /// Raises RuntimeError if the input is invalid.
    #[staticmethod]
    #[pyo3(signature = (batch_data, num_samples, sample_size, num_qubits))]
    fn validate_batch(
        batch_data: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
    ) -> PyResult<()> {
        let values = flatten_batch(&batch_data, num_samples, sample_size)?;
        CorePreprocessor::validate_batch(&values, num_samples, sample_size, num_qubits)
            .map_err(to_py_err)
    }

    /// Return the L2 norm of every sample in a batch as a float64 array.
    ///
    /// Raises RuntimeError if the shape is inconsistent or any sample has
    /// a zero, NaN or infinite norm.
    #[staticmethod]
    #[pyo3(signature = (batch_data, num_samples, sample_size))]
    fn calculate_batch_l2_norms<'py>(
        py: Python<'py>,
        batch_data: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
        num_samples: usize,
        sample_size: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = flatten_batch(&batch_data, num_samples, sample_size)?;
        // The core helper chunks by `sample_size` without checking the
        // length; `par_chunks(0)` panics, and the release profile aborts on
        // panic, so guard here instead of taking the interpreter down.
        let expected = num_samples.checked_mul(sample_size);
        if sample_size == 0 || num_samples == 0 || expected != Some(values.len()) {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid input: batch_data length {} doesn't match num_samples {} * \
                 sample_size {} (both must be > 0)",
                values.len(),
                num_samples,
                sample_size
            )));
        }
        let norms = CorePreprocessor::calculate_batch_l2_norms(&values, num_samples, sample_size)
            .map_err(to_py_err)?;
        Ok(norms.into_pyarray(py))
    }
}
