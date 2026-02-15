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
use qdp_core::dlpack::{DL_FLOAT, DLDeviceType, DLManagedTensor};
use std::ffi::c_void;

/// DLPack tensor information extracted from a PyCapsule
///
/// This struct owns the DLManagedTensor pointer and ensures proper cleanup
/// via the DLPack deleter when dropped (RAII pattern).
pub struct DLPackTensorInfo {
    /// Raw DLManagedTensor pointer from PyTorch DLPack capsule
    /// This is owned by this struct and will be freed via deleter on drop
    pub managed_ptr: *mut DLManagedTensor,
    /// Data pointer inside dl_tensor (GPU memory, owned by managed_ptr)
    pub data_ptr: *const c_void,
    pub shape: Vec<i64>,
    /// CUDA device ID from DLPack metadata.
    /// Used for defensive validation against PyTorch API device ID.
    pub device_id: i32,
}

impl Drop for DLPackTensorInfo {
    fn drop(&mut self) {
        unsafe {
            if !self.managed_ptr.is_null() {
                // Per DLPack protocol: consumer must call deleter exactly once
                if let Some(deleter) = (*self.managed_ptr).deleter {
                    deleter(self.managed_ptr);
                }
                // Prevent double-free
                self.managed_ptr = std::ptr::null_mut();
            }
        }
    }
}

/// Extract GPU pointer from PyTorch tensor's __dlpack__() capsule
///
/// Uses the DLPack protocol to obtain a zero-copy view of the tensor's GPU memory.
/// The returned `DLPackTensorInfo` owns the DLManagedTensor and will automatically
/// call the deleter when dropped, ensuring proper resource cleanup.
///
/// # Safety
/// The returned `data_ptr` points to GPU memory owned by the source tensor.
/// The caller must ensure the source tensor remains alive and unmodified
/// for the entire duration that `data_ptr` is in use. Python's GIL ensures
/// the tensor won't be garbage collected during `encode()`, but the caller
/// must not deallocate or resize the tensor while encoding is in progress.
pub fn extract_dlpack_tensor(
    _py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
) -> PyResult<DLPackTensorInfo> {
    // Call tensor.__dlpack__() to get PyCapsule
    // Note: PyTorch's __dlpack__ uses the default stream when called without arguments
    let capsule = tensor.call_method0("__dlpack__")?;

    const DLTENSOR_NAME: &[u8] = b"dltensor\0";

    // SAFETY: capsule is a valid PyCapsule from tensor.__dlpack__(). DLTENSOR_NAME is a
    // null-terminated C string for the lifetime of the call. We only read the capsule
    // and call PyCapsule_IsValid / PyCapsule_GetPointer; we do not invalidate the capsule.
    let managed_ptr = unsafe {
        let capsule_ptr = capsule.as_ptr();
        if ffi::PyCapsule_IsValid(capsule_ptr, DLTENSOR_NAME.as_ptr() as *const i8) == 0 {
            return Err(PyRuntimeError::new_err(
                "Invalid DLPack capsule (expected 'dltensor')",
            ));
        }
        let ptr = ffi::PyCapsule_GetPointer(capsule_ptr, DLTENSOR_NAME.as_ptr() as *const i8)
            as *mut DLManagedTensor;
        if ptr.is_null() {
            return Err(PyRuntimeError::new_err(
                "Failed to extract DLManagedTensor from PyCapsule",
            ));
        }
        ptr
    };

    // SAFETY: managed_ptr is non-null and was returned by PyCapsule_GetPointer for a valid
    // "dltensor" capsule, so it points to a valid DLManagedTensor. The capsule (and thus
    // the tensor) is held by the caller for the duration of this function. We read fields
    // and create slices from shape/strides only when non-null and ndim is valid.
    unsafe {
        let dl_tensor = &(*managed_ptr).dl_tensor;

        if dl_tensor.data.is_null() {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor has null data pointer",
            ));
        }

        if dl_tensor.device.device_type != DLDeviceType::kDLCUDA {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor must be on CUDA device",
            ));
        }

        if dl_tensor.dtype.code != DL_FLOAT
            || dl_tensor.dtype.bits != 64
            || dl_tensor.dtype.lanes != 1
        {
            return Err(PyRuntimeError::new_err(format!(
                "DLPack tensor must be float64 (code={}, bits={}, lanes={})",
                dl_tensor.dtype.code, dl_tensor.dtype.bits, dl_tensor.dtype.lanes
            )));
        }

        if !dl_tensor
            .byte_offset
            .is_multiple_of(std::mem::size_of::<f64>() as u64)
        {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor byte_offset is not aligned for float64",
            ));
        }

        let data_ptr =
            (dl_tensor.data as *const u8).add(dl_tensor.byte_offset as usize) as *const f64;

        let ndim = dl_tensor.ndim as usize;
        // SAFETY: shape pointer is valid for ndim elements when non-null (DLPack contract).
        let shape = if ndim > 0 && !dl_tensor.shape.is_null() {
            std::slice::from_raw_parts(dl_tensor.shape, ndim).to_vec()
        } else {
            vec![]
        };

        if ndim == 0 || shape.is_empty() {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor must have at least 1 dimension",
            ));
        }

        if !dl_tensor.strides.is_null() {
            // SAFETY: strides pointer is valid for ndim elements (DLPack contract).
            let strides = std::slice::from_raw_parts(dl_tensor.strides, ndim);
            match ndim {
                1 => {
                    let expected = 1_i64;
                    if strides[0] != expected {
                        return Err(PyRuntimeError::new_err(format!(
                            "DLPack tensor must be contiguous: stride[0]={}, expected {}",
                            strides[0], expected
                        )));
                    }
                }
                2 => {
                    if shape.len() < 2 {
                        return Err(PyRuntimeError::new_err(
                            "DLPack tensor must be contiguous (shape len < 2)",
                        ));
                    }
                    let expected_stride_1 = 1_i64;
                    let expected_stride_0 = shape[1];
                    if strides[1] != expected_stride_1 || strides[0] != expected_stride_0 {
                        return Err(PyRuntimeError::new_err(format!(
                            "DLPack tensor must be contiguous: strides=[{}, {}], expected [{}, {}] (expected[1]=shape[1])",
                            strides[0], strides[1], expected_stride_0, expected_stride_1
                        )));
                    }
                }
                _ => {
                    return Err(PyRuntimeError::new_err(
                        "DLPack tensor must be 1D or 2D for encoding",
                    ));
                }
            }
        }

        let device_id = dl_tensor.device.device_id;

        const USED_DLTENSOR_NAME: &[u8] = b"used_dltensor\0";
        // SAFETY: capsule is the same PyCapsule we used above; renaming is allowed and does not free it.
        ffi::PyCapsule_SetName(capsule.as_ptr(), USED_DLTENSOR_NAME.as_ptr() as *const i8);

        Ok(DLPackTensorInfo {
            managed_ptr,
            data_ptr: data_ptr as *const std::ffi::c_void,
            shape,
            device_id,
        })
    }
}
