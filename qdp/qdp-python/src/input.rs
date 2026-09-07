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

//! Turn a Python object into an engine [`Input`].
//!
//! Accepted objects: a Python list of floats, a NumPy array (`float64` or
//! `float32`), a CPU PyTorch tensor (viewed as NumPy), or a CUDA PyTorch
//! tensor (used in place on its current stream). One-dimensional data is a
//! single sample; two-dimensional data is `(batch, sample_size)`.

use std::ffi::c_void;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use qdp_core::{DeviceDtype, DeviceInput, Encoding, HostInput, Input, Shape};

use crate::pytorch::{
    extract_cuda_tensor_info, get_torch_cuda_stream_ptr, is_cuda_tensor, is_pytorch_tensor,
    validate_cuda_tensor_for_encoding, validate_shape, validate_tensor_cpu,
};

/// Owns whatever backs the input so the borrowed [`Input`] stays valid.
pub enum PyInput<'py> {
    List(Vec<f64>),
    F64x1(PyReadonlyArray1<'py, f64>),
    F64x2(PyReadonlyArray2<'py, f64>),
    F32x1(PyReadonlyArray1<'py, f32>),
    F32x2(PyReadonlyArray2<'py, f32>),
    Device {
        ptr: DeviceInput,
        stream: *mut c_void,
        shape: Shape,
    },
}

impl<'py> PyInput<'py> {
    pub fn from_py(
        data: &Bound<'py, PyAny>,
        encoding: Encoding,
        engine_device: usize,
    ) -> PyResult<Self> {
        if is_pytorch_tensor(data)? {
            if is_cuda_tensor(data)? {
                return Self::from_cuda_tensor(data, encoding, engine_device);
            }
            validate_tensor_cpu(data)?;
            let view = data
                .call_method0("detach")?
                .call_method0("numpy")
                .map_err(|_| {
                    PyRuntimeError::new_err(
                        "Failed to convert torch.Tensor to NumPy view. Ensure the tensor is on CPU \
                         and does not require grad (try: tensor = tensor.detach().cpu())",
                    )
                })?;
            return Self::from_array(&view, "tensor");
        }
        if data.hasattr("__array_interface__")? {
            return Self::from_array(data, "array");
        }
        let list = data.extract::<Vec<f64>>().map_err(|_| {
            PyRuntimeError::new_err(
                "Unsupported data type. Expected: list, NumPy array, PyTorch tensor, or file path",
            )
        })?;
        Ok(Self::List(list))
    }

    fn from_array(data: &Bound<'py, PyAny>, context: &str) -> PyResult<Self> {
        let ndim: usize = data.getattr("ndim")?.extract()?;
        validate_shape(ndim, context)?;
        let contiguous = |_: ()| {
            PyRuntimeError::new_err(format!(
                "{} must be contiguous (C-order) (try: .copy() or .contiguous())",
                if context == "array" {
                    "NumPy array"
                } else {
                    "Tensor"
                }
            ))
        };
        let dtype_err = || {
            PyRuntimeError::new_err(format!(
                "Failed to extract {}. Ensure dtype is float64 or float32.",
                context
            ))
        };
        match ndim {
            1 => {
                if let Ok(a) = data.extract::<PyReadonlyArray1<f64>>() {
                    a.as_slice().map_err(|_| contiguous(()))?;
                    return Ok(Self::F64x1(a));
                }
                let a = data
                    .extract::<PyReadonlyArray1<f32>>()
                    .map_err(|_| dtype_err())?;
                a.as_slice().map_err(|_| contiguous(()))?;
                Ok(Self::F32x1(a))
            }
            _ => {
                if let Ok(a) = data.extract::<PyReadonlyArray2<f64>>() {
                    a.as_slice().map_err(|_| contiguous(()))?;
                    return Ok(Self::F64x2(a));
                }
                let a = data
                    .extract::<PyReadonlyArray2<f32>>()
                    .map_err(|_| dtype_err())?;
                a.as_slice().map_err(|_| contiguous(()))?;
                Ok(Self::F32x2(a))
            }
        }
    }

    fn from_cuda_tensor(
        data: &Bound<'py, PyAny>,
        encoding: Encoding,
        engine_device: usize,
    ) -> PyResult<Self> {
        let dtype = validate_cuda_tensor_for_encoding(data, engine_device, encoding)?;
        let ndim: usize = data.call_method0("dim")?.extract()?;
        validate_shape(ndim, "CUDA tensor")?;
        let info = extract_cuda_tensor_info(data)?;
        let stream = get_torch_cuda_stream_ptr(data)?;
        let shape = match info.shape.as_slice() {
            [len] => Shape::new(1, *len as usize),
            [rows, cols] => Shape::new(*rows as usize, *cols as usize),
            _ => unreachable!("validate_shape() guarantees ndim is 1 or 2"),
        };
        let raw = info.data_ptr as *const c_void;
        let ptr = match dtype {
            DeviceDtype::F64 => DeviceInput::F64(raw as *const f64),
            DeviceDtype::F32 => DeviceInput::F32(raw as *const f32),
            DeviceDtype::I64 => DeviceInput::I64(raw as *const usize),
        };
        Ok(Self::Device { ptr, stream, shape })
    }

    /// The engine input and its batch geometry.
    pub fn as_input(&self) -> PyResult<(Input<'_>, Shape)> {
        fn shape1(len: usize) -> Shape {
            Shape::new(1, len)
        }
        fn shape2(dims: &[usize]) -> Shape {
            Shape::new(dims[0], dims[1])
        }
        let slice_err = |_| PyRuntimeError::new_err("array must be contiguous (C-order)");
        Ok(match self {
            Self::List(v) => (Input::Host(HostInput::F64(v)), shape1(v.len())),
            Self::F64x1(a) => {
                let s = a.as_slice().map_err(slice_err)?;
                (Input::Host(HostInput::F64(s)), shape1(s.len()))
            }
            Self::F32x1(a) => {
                let s = a.as_slice().map_err(slice_err)?;
                (Input::Host(HostInput::F32(s)), shape1(s.len()))
            }
            Self::F64x2(a) => (
                Input::Host(HostInput::F64(a.as_slice().map_err(slice_err)?)),
                shape2(a.shape()),
            ),
            Self::F32x2(a) => (
                Input::Host(HostInput::F32(a.as_slice().map_err(slice_err)?)),
                shape2(a.shape()),
            ),
            Self::Device { ptr, stream, shape } => (
                Input::Device {
                    ptr: *ptr,
                    stream: *stream,
                },
                *shape,
            ),
        })
    }
}
