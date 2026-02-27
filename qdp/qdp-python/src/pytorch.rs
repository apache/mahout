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
use pyo3::prelude::*;
use std::ffi::c_void;

/// Helper to detect PyTorch tensor
pub fn is_pytorch_tensor(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
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
pub fn validate_tensor(tensor: &Bound<'_, PyAny>) -> PyResult<()> {
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
pub fn is_cuda_tensor(tensor: &Bound<'_, PyAny>) -> PyResult<bool> {
    let device = tensor.getattr("device")?;
    let device_type: String = device.getattr("type")?.extract()?;
    Ok(device_type == "cuda")
}

const CUDA_ENCODING_METHODS: &[&str] = &["amplitude", "angle", "basis", "iqp", "iqp-z"];

fn format_supported_cuda_encoding_methods() -> String {
    let quoted: Vec<String> = CUDA_ENCODING_METHODS
        .iter()
        .map(|method| format!("'{}'", method))
        .collect();
    let len = quoted.len();
    if len == 1 {
        return quoted[0].clone();
    }
    format!("{}, or {}", quoted[..len - 1].join(", "), quoted[len - 1])
}

/// Validate array/tensor shape (must be 1D or 2D)
///
/// Args:
///     ndim: Number of dimensions
///     context: Context string for error message (e.g., "array", "tensor", "CUDA tensor")
///
/// Returns:
///     Ok(()) if shape is valid (1D or 2D), otherwise returns an error
pub fn validate_shape(ndim: usize, context: &str) -> PyResult<()> {
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
pub fn get_tensor_device_id(tensor: &Bound<'_, PyAny>) -> PyResult<i32> {
    let device = tensor.getattr("device")?;
    let device_index: i32 = device.getattr("index")?.extract()?;
    Ok(device_index)
}

/// Get the current CUDA stream pointer for the tensor's device.
pub fn get_torch_cuda_stream_ptr(tensor: &Bound<'_, PyAny>) -> PyResult<*mut c_void> {
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

    let stream_ptr: u64 = stream.getattr("cuda_stream")?.extract()?;
    Ok(if stream_ptr == 0 {
        std::ptr::null_mut()
    } else {
        stream_ptr as *mut c_void
    })
}

/// Validate a CUDA tensor for direct GPU encoding
/// Checks: dtype matches encoding method, contiguous, non-empty, device_id matches engine
pub fn validate_cuda_tensor_for_encoding(
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
        "amplitude" => {
            if !(dtype_str_lower.contains("float64") || dtype_str_lower.contains("float32")) {
                return Err(PyRuntimeError::new_err(format!(
                    "CUDA tensor must have dtype float64 or float32 for amplitude encoding, got {}. \
                     Use tensor.to(torch.float64) or tensor.to(torch.float32)",
                    dtype_str
                )));
            }
        }
        "angle" | "iqp" | "iqp-z" => {
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
                "CUDA tensor encoding currently only supports {} methods, got '{}'. \
                 Use tensor.cpu() to convert to CPU tensor for other encoding methods.",
                format_supported_cuda_encoding_methods(),
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
pub struct CudaTensorInfo {
    pub data_ptr: *const f64,
    pub shape: Vec<i64>,
}

/// Extract GPU pointer and shape directly from a PyTorch CUDA tensor.
///
/// # Safety
/// The returned pointer is borrowed from the source tensor. The caller must
/// ensure the tensor remains alive and unmodified for the duration of use.
pub fn extract_cuda_tensor_info(tensor: &Bound<'_, PyAny>) -> PyResult<CudaTensorInfo> {
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
