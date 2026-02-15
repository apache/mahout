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

use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
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
