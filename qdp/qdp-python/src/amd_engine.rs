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

use crate::dlpack::steal_dlpack_managed_tensor;
use crate::tensor::QuantumTensor;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use qdp_core::Precision;

#[pyclass]
pub struct AmdQdpEngine {
    device_id: usize,
    precision: Precision,
}

impl AmdQdpEngine {
    fn torch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
        PyModule::import(py, "torch")
            .map_err(|_| PyRuntimeError::new_err("Failed to import torch module"))
    }

    fn device_str(&self) -> String {
        format!("cuda:{}", self.device_id)
    }

    fn real_dtype_attr<'py>(&self, torch: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyAny>> {
        match self.precision {
            Precision::Float32 => torch.getattr("float32"),
            Precision::Float64 => torch.getattr("float64"),
        }
    }

    fn complex_dtype_attr<'py>(&self, torch: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyAny>> {
        match self.precision {
            Precision::Float32 => torch.getattr("complex64"),
            Precision::Float64 => torch.getattr("complex128"),
        }
    }

    fn to_device_tensor<'py>(
        &self,
        torch: &Bound<'py, PyModule>,
        data: &Bound<'py, PyAny>,
        dtype: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(data.py());
        kwargs.set_item("device", self.device_str())?;
        if let Some(dtype) = dtype {
            kwargs.set_item("dtype", dtype)?;
        }
        torch.call_method("as_tensor", (data,), Some(&kwargs))
    }

    fn validate_rocm_runtime(&self, py: Python<'_>) -> PyResult<()> {
        let torch = self.torch(py)?;
        let version = torch.getattr("version")?;
        let hip = version.getattr("hip")?;
        if hip.is_none() {
            return Err(PyRuntimeError::new_err(
                "AMD backend requires PyTorch ROCm build (torch.version.hip is None).",
            ));
        }
        Ok(())
    }

    fn encode_amplitude<'py>(
        &self,
        torch: &Bound<'py, PyModule>,
        data: &Bound<'py, PyAny>,
        num_qubits: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let real_dtype = self.real_dtype_attr(torch)?;
        let x = self.to_device_tensor(torch, data, Some(real_dtype.clone()))?;
        let ndim: usize = x.call_method0("dim")?.extract()?;
        if ndim != 1 && ndim != 2 {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported tensor shape: {}D. Expected 1D or 2D.",
                ndim
            )));
        }

        let x2d = if ndim == 1 {
            x.call_method1("unsqueeze", (0,))?
        } else {
            x
        };

        let shape = x2d.getattr("shape")?;
        let sample_size: usize = shape.get_item(1)?.extract()?;
        let state_len = 1usize << num_qubits;
        if sample_size != state_len {
            return Err(PyRuntimeError::new_err(format!(
                "Amplitude encoding expects sample size {} (=2^num_qubits), got {}",
                state_len, sample_size
            )));
        }

        let norm_kwargs = PyDict::new(data.py());
        norm_kwargs.set_item("dim", -1)?;
        norm_kwargs.set_item("keepdim", true)?;
        let norms = torch.call_method("norm", (&x2d,), Some(&norm_kwargs))?;
        let clamp_kwargs = PyDict::new(data.py());
        clamp_kwargs.set_item("min", 1e-12_f64)?;
        let norms = torch.call_method("clamp", (&norms,), Some(&clamp_kwargs))?;
        let normalized = x2d.call_method1("__truediv__", (&norms,))?;

        let imag = torch.call_method1("zeros_like", (&normalized,))?;
        let complex_state = torch.call_method1("complex", (&normalized, &imag))?;
        complex_state.call_method1("to", (self.complex_dtype_attr(torch)?,))
    }

    fn encode_basis<'py>(
        &self,
        torch: &Bound<'py, PyModule>,
        data: &Bound<'py, PyAny>,
        num_qubits: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let indices = self.to_device_tensor(torch, data, Some(torch.getattr("int64")?))?;
        let ndim: usize = indices.call_method0("dim")?.extract()?;
        if ndim != 1 && ndim != 2 {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported tensor shape: {}D. Expected 1D or 2D.",
                ndim
            )));
        }

        let indices_1d = if ndim == 2 {
            let shape = indices.getattr("shape")?;
            let width: usize = shape.get_item(1)?.extract()?;
            if width != 1 {
                return Err(PyRuntimeError::new_err(format!(
                    "Basis encoding expects 2D input width 1, got {}",
                    width
                )));
            }
            indices.call_method1("squeeze", (1,))?
        } else {
            indices
        };

        let batch: usize = indices_1d.call_method0("numel")?.extract()?;
        if batch == 0 {
            return Err(PyRuntimeError::new_err("Basis tensor cannot be empty"));
        }

        let state_len = 1usize << num_qubits;
        let min_idx: i64 = indices_1d.call_method0("min")?.extract()?;
        let max_idx: i64 = indices_1d.call_method0("max")?.extract()?;
        if min_idx < 0 || max_idx >= state_len as i64 {
            return Err(PyRuntimeError::new_err(format!(
                "Basis index out of range. Valid range: [0, {}], got min={}, max={}",
                state_len.saturating_sub(1),
                min_idx,
                max_idx
            )));
        }

        let state_shape = (batch, state_len);
        let zeros_kwargs = PyDict::new(data.py());
        zeros_kwargs.set_item("device", self.device_str())?;
        zeros_kwargs.set_item("dtype", self.complex_dtype_attr(torch)?)?;
        let states = torch.call_method("zeros", (state_shape,), Some(&zeros_kwargs))?;

        let one_shape = (batch, 1usize);
        let ones_kwargs = PyDict::new(data.py());
        ones_kwargs.set_item("device", self.device_str())?;
        ones_kwargs.set_item("dtype", self.complex_dtype_attr(torch)?)?;
        let ones = torch.call_method("ones", (one_shape,), Some(&ones_kwargs))?;
        let index_2d = indices_1d.call_method1("reshape", (one_shape,))?;
        states.call_method1("scatter_", (1, index_2d, ones))?;
        Ok(states)
    }

    fn encode_angle<'py>(
        &self,
        torch: &Bound<'py, PyModule>,
        data: &Bound<'py, PyAny>,
        num_qubits: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let real_dtype = self.real_dtype_attr(torch)?;
        let angles = self.to_device_tensor(torch, data, Some(real_dtype.clone()))?;
        let ndim: usize = angles.call_method0("dim")?.extract()?;
        if ndim != 1 && ndim != 2 {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported tensor shape: {}D. Expected 1D or 2D.",
                ndim
            )));
        }
        let angles2d = if ndim == 1 {
            angles.call_method1("unsqueeze", (0,))?
        } else {
            angles
        };

        let shape = angles2d.getattr("shape")?;
        let width: usize = shape.get_item(1)?.extract()?;
        if width != num_qubits {
            return Err(PyRuntimeError::new_err(format!(
                "Angle encoding expects {} features (=num_qubits), got {}",
                num_qubits, width
            )));
        }

        let batch: usize = shape.get_item(0)?.extract()?;
        let state_len = 1usize << num_qubits;
        let one_shape = (batch, state_len);
        let ones_kwargs = PyDict::new(data.py());
        ones_kwargs.set_item("device", self.device_str())?;
        ones_kwargs.set_item("dtype", real_dtype)?;
        let mut amplitudes = torch.call_method("ones", (one_shape,), Some(&ones_kwargs))?;

        let idx_kwargs = PyDict::new(data.py());
        idx_kwargs.set_item("device", self.device_str())?;
        let idx = torch
            .call_method("arange", (state_len,), Some(&idx_kwargs))?
            .call_method1("reshape", ((1usize, state_len),))?;

        for bit in 0..num_qubits {
            let shifted = idx.call_method1("__rshift__", (bit,))?;
            let bitvals = shifted.call_method1("__and__", (1,))?;
            let mask = bitvals.call_method1("eq", (1,))?;

            let angle_col = angles2d
                .call_method1("select", (1, bit))?
                .call_method1("unsqueeze", (1,))?;
            let sin_term = torch.call_method1("sin", (&angle_col,))?;
            let cos_term = torch.call_method1("cos", (&angle_col,))?;
            let factor = torch.call_method1("where", (&mask, &sin_term, &cos_term))?;
            amplitudes = amplitudes.call_method1("__mul__", (&factor,))?;
        }

        let imag = torch.call_method1("zeros_like", (&amplitudes,))?;
        let complex_state = torch.call_method1("complex", (&amplitudes, &imag))?;
        complex_state.call_method1("to", (self.complex_dtype_attr(torch)?,))
    }
}

#[pymethods]
impl AmdQdpEngine {
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
        Ok(Self {
            device_id,
            precision,
        })
    }

    #[pyo3(signature = (data, num_qubits, encoding_method="amplitude"))]
    fn encode(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        self.validate_rocm_runtime(py)?;
        let torch = self.torch(py)?;
        let method = encoding_method.to_ascii_lowercase();
        let encoded = match method.as_str() {
            "amplitude" => self.encode_amplitude(&torch, data, num_qubits)?,
            "angle" => self.encode_angle(&torch, data, num_qubits)?,
            "basis" => self.encode_basis(&torch, data, num_qubits)?,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "AMD backend currently supports 'amplitude', 'angle', and 'basis'. Got '{}'.",
                    encoding_method
                )));
            }
        };

        let encoded = encoded.call_method0("contiguous")?;
        let ptr = steal_dlpack_managed_tensor(&encoded)?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    fn backend(&self) -> PyResult<&str> {
        Ok("amd-rocm")
    }

    fn device(&self) -> PyResult<String> {
        Ok(self.device_str())
    }

    fn precision(&self) -> PyResult<&str> {
        match self.precision {
            Precision::Float32 => Ok("float32"),
            Precision::Float64 => Ok("float64"),
        }
    }
}
