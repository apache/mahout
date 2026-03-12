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

use numpy::{PyArray2, ndarray::Array2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use qdp_core::dlpack::DLManagedTensor;
use std::ffi::c_void;

// CUDA Runtime API — already linked transitively by qdp-core.
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

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
pub struct QuantumTensor {
    pub ptr: *mut DLManagedTensor,
    pub consumed: bool,
}

#[pymethods]
impl QuantumTensor {
    /// Implements DLPack protocol - returns PyCapsule for PyTorch
    ///
    /// This method is called by torch.from_dlpack() to get the GPU memory pointer.
    /// The capsule can only be consumed once to prevent double-free errors.
    ///
    /// Args:
    ///     stream: Optional CUDA stream (DLPack 0.8+; 1=legacy default, 2=per-thread default)
    ///
    /// Returns:
    ///     PyCapsule containing DLManagedTensor pointer
    ///
    /// Raises:
    ///     RuntimeError: If the tensor has already been consumed
    #[pyo3(signature = (stream=None))]
    fn __dlpack__<'py>(&mut self, py: Python<'py>, stream: Option<i64>) -> PyResult<Py<PyAny>> {
        if self.consumed {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor already consumed (can only be used once)",
            ));
        }

        if self.ptr.is_null() {
            return Err(PyRuntimeError::new_err("Invalid DLPack tensor pointer"));
        }

        if let Some(stream) = stream
            && stream > 0
        {
            let stream_ptr = qdp_core::dlpack::dlpack_stream_to_cuda(stream);
            unsafe {
                qdp_core::dlpack::synchronize_stream(stream_ptr).map_err(|e| {
                    PyRuntimeError::new_err(format!("CUDA stream sync failed: {}", e))
                })?;
            }
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

    /// Copy encoded quantum state from GPU to a NumPy array (CPU, float64).
    ///
    /// Performs a synchronous cudaMemcpy D2H without requiring PyTorch.
    /// Complex128 output (imaginary parts are always 0.0 per the CUDA kernel)
    /// is reduced to float64 by discarding the zero imaginary components.
    ///
    /// Returns:
    ///     numpy.ndarray of shape (batch_size, state_len), dtype float64.
    ///
    /// Raises:
    ///     RuntimeError: If the tensor has already been consumed, the pointer is
    ///                   invalid, the dtype is unsupported, or the CUDA copy fails.
    #[allow(clippy::wrong_self_convention)] // mut required: sets self.consumed and calls DLPack deleter
    fn to_numpy<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.consumed {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor already consumed (can only be used once)",
            ));
        }
        if self.ptr.is_null() {
            return Err(PyRuntimeError::new_err("Invalid DLPack tensor pointer"));
        }

        let (rows, cols, host_data) = unsafe {
            let dl_tensor = &(*self.ptr).dl_tensor;

            // Shape — require 1-D or 2-D.
            let ndim = dl_tensor.ndim as usize;
            if ndim == 0 || ndim > 2 || dl_tensor.shape.is_null() {
                return Err(PyRuntimeError::new_err(
                    "to_numpy() requires a 1-D or 2-D tensor",
                ));
            }
            let shape = std::slice::from_raw_parts(dl_tensor.shape, ndim);
            let (rows, cols) = if ndim == 1 {
                (1usize, shape[0] as usize)
            } else {
                (shape[0] as usize, shape[1] as usize)
            };

            // Dtype: complex128 (DL_COMPLEX=5, bits=128) or float64 (DL_FLOAT=2, bits=64).
            let dtype = &dl_tensor.dtype;
            let (is_complex, elem_bytes) = match (dtype.code, dtype.bits) {
                (5, 128) => (true, 16usize),
                (2, 64) => (false, 8usize),
                _ => {
                    return Err(PyRuntimeError::new_err(format!(
                        "to_numpy() unsupported dtype: code={}, bits={}",
                        dtype.code, dtype.bits
                    )));
                }
            };

            let n_elems = rows * cols;
            // For complex128 each element is two consecutive f64 values.
            let host_f64_count = if is_complex { n_elems * 2 } else { n_elems };
            let mut host_buf = vec![0.0f64; host_f64_count];

            let data_ptr = (dl_tensor.data as *const u8).add(dl_tensor.byte_offset as usize);

            let ret = cudaMemcpy(
                host_buf.as_mut_ptr() as *mut c_void,
                data_ptr as *const c_void,
                n_elems * elem_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            if ret != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMemcpy D2H failed with error code {}",
                    ret
                )));
            }

            // Consumed: GPU memory is ours to free now.
            self.consumed = true;
            if let Some(deleter) = (*self.ptr).deleter {
                deleter(self.ptr);
            }

            // complex128 → float64: discard imaginary parts (always 0.0).
            let host_data: Vec<f64> = if is_complex {
                host_buf.into_iter().step_by(2).collect()
            } else {
                host_buf
            };

            (rows, cols, host_data)
        };

        let arr = Array2::from_shape_vec((rows, cols), host_data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
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
            // DLPack device_type: kDLCUDA = 2, kDLCPU = 1
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
        // Only free if not consumed; __dlpack__ leaves freeing to PyTorch,
        // to_numpy() calls the deleter itself after the D2H copy.
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
