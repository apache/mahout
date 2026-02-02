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

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::ffi;
use pyo3::prelude::*;
use qdp_core::dlpack::{DL_FLOAT, DLDeviceType, DLManagedTensor};
use qdp_core::{Precision, QdpEngine as CoreEngine};
use std::ffi::c_void;

#[cfg(target_os = "linux")]
use qdp_core::{PipelineConfig, PipelineIterator, PipelineRunResult, run_throughput_pipeline};

/// Wraps raw DLPack pointer so it can cross `py.allow_threads` (closure return must be `Send`).
/// Safe: DLPack pointer handover across contexts; GIL is released only during the closure.
struct SendPtr(pub *mut DLManagedTensor);
unsafe impl Send for SendPtr {}

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

/// Python iterator yielding one QuantumTensor (batch) per __next__. Releases GIL during next_batch().
#[cfg(target_os = "linux")]
#[pyclass]
struct PyQuantumLoader {
    inner: Option<PipelineIterator>,
}

#[cfg(target_os = "linux")]
#[pymethods]
impl PyQuantumLoader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Returns the next batch as QuantumTensor; raises StopIteration when exhausted. Releases GIL during encode.
    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<QuantumTensor> {
        let mut inner_iter = match slf.inner.take() {
            Some(it) => it,
            None => return Err(PyStopIteration::new_err("loader exhausted")),
        };

        #[allow(deprecated)]
        let result = py.allow_threads(move || {
            let res = inner_iter.next_batch();
            match res {
                Ok(Some(ptr)) => Ok((inner_iter, Some(SendPtr(ptr)))),
                Ok(None) => Ok((inner_iter, None)),
                Err(e) => Err((inner_iter, e)),
            }
        });

        match result {
            Ok((returned_iter, Some(send_ptr))) => {
                slf.inner = Some(returned_iter);
                Ok(QuantumTensor {
                    ptr: send_ptr.0,
                    consumed: false,
                })
            }
            Ok((_, None)) => Err(PyStopIteration::new_err("loader exhausted")),
            Err((returned_iter, e)) => {
                slf.inner = Some(returned_iter);
                Err(PyRuntimeError::new_err(e.to_string()))
            }
        }
    }
}

/// Stub PyQuantumLoader when not on Linux (CUDA pipeline not available).
#[cfg(not(target_os = "linux"))]
#[pyclass]
struct PyQuantumLoader {}

#[cfg(not(target_os = "linux"))]
#[pymethods]
impl PyQuantumLoader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, _py: Python<'_>) -> PyResult<QuantumTensor> {
        Err(PyRuntimeError::new_err(
            "QuantumDataLoader is only available on Linux (CUDA pipeline). \
             Build and run from a Linux host with CUDA.",
        ))
    }
}

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

/// Validate array/tensor shape (must be 1D or 2D)
///
/// Args:
///     ndim: Number of dimensions
///     context: Context string for error message (e.g., "array", "tensor", "CUDA tensor")
///
/// Returns:
///     Ok(()) if shape is valid (1D or 2D), otherwise returns an error
fn validate_shape(ndim: usize, context: &str) -> PyResult<()> {
    match ndim {
        1 | 2 => Ok(()),
        _ => {
            let item_type = if context.contains("array") {
                "array"
            } else {
                "tensor"
            };
            Err(PyRuntimeError::new_err(format!(
                "Unsupported {} shape: {}D. Expected 1D {} for single sample \
                 encoding or 2D {} (batch_size, features) for batch encoding.",
                context, ndim, item_type, item_type
            )))
        }
    }
}

/// Get the CUDA device index from a PyTorch tensor
fn get_tensor_device_id(tensor: &Bound<'_, PyAny>) -> PyResult<i32> {
    let device = tensor.getattr("device")?;
    let device_index: i32 = device.getattr("index")?.extract()?;
    Ok(device_index)
}

/// Get the current CUDA stream pointer for the tensor's device.
fn get_torch_cuda_stream_ptr(tensor: &Bound<'_, PyAny>) -> PyResult<*mut c_void> {
    let py = tensor.py();
    let torch = PyModule::import(py, "torch")
        .map_err(|_| PyRuntimeError::new_err("Failed to import torch module"))?;
    let cuda = torch.getattr("cuda")?;
    let device = tensor.getattr("device")?;
    let stream = cuda.call_method1("current_stream", (device,))?;

    // Defensive validation: ensure the stream is a CUDA stream on the same device
    let stream_device = stream.getattr("device").map_err(|_| {
        PyRuntimeError::new_err("CUDA stream object from PyTorch is missing 'device' attribute")
    })?;
    let stream_device_type: String = stream_device
        .getattr("type")
        .and_then(|obj| obj.extract())
        .map_err(|_| {
            PyRuntimeError::new_err(
                "Failed to extract CUDA stream device type from PyTorch stream.device",
            )
        })?;
    if stream_device_type != "cuda" {
        return Err(PyRuntimeError::new_err(format!(
            "Expected CUDA stream device type 'cuda', got '{}'",
            stream_device_type
        )));
    }

    let stream_device_index: i32 = stream_device
        .getattr("index")
        .and_then(|obj| obj.extract())
        .map_err(|_| {
            PyRuntimeError::new_err(
                "Failed to extract CUDA stream device index from PyTorch stream.device",
            )
        })?;
    let tensor_device_index = get_tensor_device_id(tensor)?;
    if stream_device_index != tensor_device_index {
        return Err(PyRuntimeError::new_err(format!(
            "CUDA stream device index ({}) does not match tensor device index ({})",
            stream_device_index, tensor_device_index
        )));
    }

    // PyTorch default stream can report cuda_stream as 0; treat as valid (Rust sync is no-op for null).
    let stream_ptr: u64 = stream.getattr("cuda_stream")?.extract()?;
    Ok(stream_ptr as *mut c_void)
}

/// Validate a CUDA tensor for direct GPU encoding
/// Checks: dtype matches encoding method, contiguous, non-empty, device_id matches engine
fn validate_cuda_tensor_for_encoding(
    tensor: &Bound<'_, PyAny>,
    expected_device_id: usize,
    encoding_method: &str,
) -> PyResult<()> {
    let method = encoding_method.to_ascii_lowercase();

    // Check encoding method support and dtype (ASCII lowercase for case-insensitive match).
    let dtype = tensor.getattr("dtype")?;
    let dtype_str: String = dtype.str()?.extract()?;
    let dtype_str_lower = dtype_str.to_ascii_lowercase();
    match method.as_str() {
        "amplitude" | "angle" => {
            if !dtype_str_lower.contains("float64") {
                return Err(PyRuntimeError::new_err(format!(
                    "CUDA tensor must have dtype float64 for {} encoding, got {}. \
                     Use tensor.to(torch.float64)",
                    method, dtype_str
                )));
            }
        }
        "basis" => {
            if !dtype_str_lower.contains("int64") {
                return Err(PyRuntimeError::new_err(format!(
                    "CUDA tensor must have dtype int64 for basis encoding, got {}. \
                     Use tensor.to(torch.int64)",
                    dtype_str
                )));
            }
        }
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "CUDA tensor encoding currently only supports 'amplitude', 'angle', or 'basis' methods, got '{}'. \
                 Use tensor.cpu() to convert to CPU tensor for other encoding methods.",
                encoding_method
            )));
        }
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

/// Minimal CUDA tensor metadata extracted via PyTorch APIs.
struct CudaTensorInfo {
    data_ptr: *const f64,
    shape: Vec<i64>,
}

/// Extract GPU pointer and shape directly from a PyTorch CUDA tensor.
///
/// # Safety
/// The returned pointer is borrowed from the source tensor. The caller must
/// ensure the tensor remains alive and unmodified for the duration of use.
fn extract_cuda_tensor_info(tensor: &Bound<'_, PyAny>) -> PyResult<CudaTensorInfo> {
    let data_ptr: u64 = tensor.call_method0("data_ptr")?.extract()?;
    if data_ptr == 0 {
        return Err(PyRuntimeError::new_err(
            "PyTorch returned a null data pointer for CUDA tensor",
        ));
    }

    let ndim: usize = tensor.call_method0("dim")?.extract()?;
    let mut shape = Vec::with_capacity(ndim);
    for axis in 0..ndim {
        let dim: i64 = tensor.call_method1("size", (axis,))?.extract()?;
        shape.push(dim);
    }

    Ok(CudaTensorInfo {
        data_ptr: data_ptr as *const f64,
        shape,
    })
}

/// DLPack tensor information extracted from a PyCapsule
///
/// This struct owns the DLManagedTensor pointer and ensures proper cleanup
/// via the DLPack deleter when dropped (RAII pattern).
struct DLPackTensorInfo {
    /// Raw DLManagedTensor pointer from PyTorch DLPack capsule
    /// This is owned by this struct and will be freed via deleter on drop
    managed_ptr: *mut DLManagedTensor,
    /// Data pointer inside dl_tensor (GPU memory, owned by managed_ptr)
    data_ptr: *const c_void,
    shape: Vec<i64>,
    /// CUDA device ID from DLPack metadata.
    /// Used for defensive validation against PyTorch API device ID.
    device_id: i32,
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
fn extract_dlpack_tensor(_py: Python<'_>, tensor: &Bound<'_, PyAny>) -> PyResult<DLPackTensorInfo> {
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
            return self.encode_from_numpy(data, num_qubits, encoding_method);
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
                let stream_ptr = get_torch_cuda_stream_ptr(data)?;

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

    /// Create a synthetic-data loader iterator for use in Python `for qt in loader`.
    ///
    /// Yields one QuantumTensor (batch) per iteration; releases GIL during encode.
    /// Use with QuantumDataLoader builder or directly for streaming encode.
    ///
    /// Args:
    ///     total_batches: Number of batches to yield
    ///     batch_size: Samples per batch
    ///     num_qubits: Qubits per sample
    ///     encoding_method: "amplitude", "angle", or "basis"
    ///     seed: Optional RNG seed for reproducible synthetic data
    ///
    /// Returns:
    ///     PyQuantumLoader: iterator yielding QuantumTensor per __next__
    #[cfg(target_os = "linux")]
    #[pyo3(signature = (total_batches, batch_size=64, num_qubits=16, encoding_method="amplitude", seed=None))]
    fn create_synthetic_loader(
        &self,
        total_batches: usize,
        batch_size: usize,
        num_qubits: u32,
        encoding_method: &str,
        seed: Option<u64>,
    ) -> PyResult<PyQuantumLoader> {
        let config = PipelineConfig {
            device_id: self.engine.device().ordinal(),
            num_qubits,
            batch_size,
            total_batches,
            encoding_method: encoding_method.to_string(),
            seed,
            warmup_batches: 0,
        };
        let iter = PipelineIterator::new_synthetic(self.engine.clone(), config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyQuantumLoader { inner: Some(iter) })
    }

    /// Stub when not on Linux: create_synthetic_loader is only implemented on Linux.
    #[cfg(not(target_os = "linux"))]
    #[pyo3(signature = (total_batches, batch_size=64, num_qubits=16, encoding_method="amplitude", seed=None))]
    fn create_synthetic_loader(
        &self,
        total_batches: usize,
        batch_size: usize,
        num_qubits: u32,
        encoding_method: &str,
        seed: Option<u64>,
    ) -> PyResult<PyQuantumLoader> {
        let _ = (total_batches, batch_size, num_qubits, encoding_method, seed);
        Err(PyRuntimeError::new_err(
            "create_synthetic_loader is only available on Linux (CUDA pipeline). \
             Build and run from a Linux host with CUDA.",
        ))
    }

    /// Run dual-stream pipeline for encoding (H2D + kernel overlap). Internal API.
    ///
    /// Exposes run_dual_stream_pipeline from qdp-core. Accepts 1D host data (single sample).
    /// Does not return a tensor; use for throughput measurement or when state is not needed.
    /// Currently supports amplitude encoding only.
    ///
    /// Args:
    ///     host_data: 1D input (list or NumPy array, float64)
    ///     num_qubits: Number of qubits
    ///     encoding_method: "amplitude" (other methods not yet supported for this path)
    #[cfg(target_os = "linux")]
    fn _encode_stream_internal(
        &self,
        host_data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<()> {
        let data_slice: Vec<f64> = if host_data.hasattr("__array_interface__")? {
            let array_1d = host_data.extract::<PyReadonlyArray1<f64>>().map_err(|_| {
                PyRuntimeError::new_err("host_data must be 1D NumPy array with dtype float64")
            })?;
            array_1d
                .as_slice()
                .map_err(|_| PyRuntimeError::new_err("NumPy array must be contiguous (C-order)"))?
                .to_vec()
        } else {
            host_data.extract::<Vec<f64>>().map_err(|_| {
                PyRuntimeError::new_err("host_data must be 1D list/array of float64")
            })?
        };
        self.engine
            .run_dual_stream_encode(&data_slice, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("run_dual_stream_encode failed: {}", e)))
    }
}

/// Runs the full throughput pipeline in Rust with GIL released. Returns (duration_sec, vectors_per_sec, latency_ms_per_vector).
#[cfg(target_os = "linux")]
#[pyfunction]
#[pyo3(signature = (device_id=0, num_qubits=16, batch_size=64, total_batches=100, encoding_method="amplitude", warmup_batches=0, seed=None))]
#[allow(clippy::too_many_arguments)]
fn run_throughput_pipeline_py_impl(
    py: Python<'_>,
    device_id: usize,
    num_qubits: u32,
    batch_size: usize,
    total_batches: usize,
    encoding_method: &str,
    warmup_batches: usize,
    seed: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    let encoding_method = encoding_method.to_string();
    #[allow(deprecated)]
    let result: Result<PipelineRunResult, qdp_core::MahoutError> = py.allow_threads(move || {
        let config = PipelineConfig {
            device_id,
            num_qubits,
            batch_size,
            total_batches,
            encoding_method,
            seed,
            warmup_batches,
        };
        run_throughput_pipeline(&config)
    });
    let res = result.map_err(|e: qdp_core::MahoutError| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        res.duration_sec,
        res.vectors_per_sec,
        res.latency_ms_per_vector,
    ))
}

/// Stub when not on Linux: run_throughput_pipeline_py is only implemented on Linux (CUDA pipeline).
#[cfg(not(target_os = "linux"))]
#[pyfunction]
#[pyo3(signature = (device_id=0, num_qubits=16, batch_size=64, total_batches=100, encoding_method="amplitude", warmup_batches=0, seed=None))]
fn run_throughput_pipeline_py_impl(
    _py: Python<'_>,
    _device_id: usize,
    _num_qubits: u32,
    _batch_size: usize,
    _total_batches: usize,
    _encoding_method: &str,
    _warmup_batches: usize,
    _seed: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    Err(PyRuntimeError::new_err(
        "run_throughput_pipeline_py is only available on Linux (CUDA pipeline). \
         Build and run from a Linux host with CUDA.",
    ))
}

/// Public wrapper so the same name is always present in the module (import never fails).
#[pyfunction]
#[pyo3(signature = (device_id=0, num_qubits=16, batch_size=64, total_batches=100, encoding_method="amplitude", warmup_batches=0, seed=None))]
#[allow(clippy::too_many_arguments)]
fn run_throughput_pipeline_py(
    py: Python<'_>,
    device_id: usize,
    num_qubits: u32,
    batch_size: usize,
    total_batches: usize,
    encoding_method: &str,
    warmup_batches: usize,
    seed: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    run_throughput_pipeline_py_impl(
        py,
        device_id,
        num_qubits,
        batch_size,
        total_batches,
        encoding_method,
        warmup_batches,
        seed,
    )
}

/// Quantum Data Plane (QDP) Python module
///
/// GPU-accelerated quantum data encoding with DLPack integration.
#[pymodule]
fn _qdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Respect RUST_LOG; try_init() is idempotent if already initialized
    let _ = env_logger::Builder::from_default_env().try_init();

    m.add_class::<QdpEngine>()?;
    m.add_class::<QuantumTensor>()?;
    m.add_class::<PyQuantumLoader>()?;
    m.add_function(pyo3::wrap_pyfunction!(run_throughput_pipeline_py, m)?)?;
    Ok(())
}
