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

//! Named shorthands over [`QdpEngine::encode`] for host batches and for
//! `f64` / `int64` device pointers addressed by encoding name.
//!
//! Every method here is a one-line rewrite into the generic call. New code
//! should call [`QdpEngine::encode`] with an [`Input`] directly.

use std::ffi::c_void;

use crate::dlpack::DLManagedTensor;
use crate::gpu::kernels::{DeviceInput, HostInput, Input, Shape};
use crate::types::Encoding;
use crate::{QdpEngine, Result};

impl QdpEngine {
    /// Encode one host sample of `f64`.
    pub fn encode_single(
        &self,
        data: &[f64],
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        self.encode(
            Input::Host(HostInput::F64(data)),
            Shape::new(1, data.len()),
            num_qubits,
            Encoding::from_str_ci(encoding_method)?,
        )
    }

    /// Encode a host batch of `f64`.
    pub fn encode_batch(
        &self,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        self.encode_batch_for_pipeline(
            batch_data,
            num_samples,
            sample_size,
            num_qubits,
            Encoding::from_str_ci(encoding_method)?,
        )
    }

    /// Encode a host batch of `f32`.
    pub fn encode_batch_f32(
        &self,
        batch_data: &[f32],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        self.encode_batch_f32_for_pipeline(
            batch_data,
            num_samples,
            sample_size,
            num_qubits,
            Encoding::from_str_ci(encoding_method)?,
        )
    }

    pub(crate) fn encode_batch_for_pipeline(
        &self,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding: Encoding,
    ) -> Result<*mut DLManagedTensor> {
        self.encode(
            Input::Host(HostInput::F64(batch_data)),
            Shape::new(num_samples, sample_size),
            num_qubits,
            encoding,
        )
    }

    pub(crate) fn encode_batch_f32_for_pipeline(
        &self,
        batch_data: &[f32],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding: Encoding,
    ) -> Result<*mut DLManagedTensor> {
        self.encode(
            Input::Host(HostInput::F32(batch_data)),
            Shape::new(num_samples, sample_size),
            num_qubits,
            encoding,
        )
    }

    /// Device pointer for `encoding_method`'s natural element type: `int64`
    /// indices for basis, `f64` for everything else.
    fn device_ptr_for(ptr: *const c_void, encoding: Encoding) -> DeviceInput {
        match encoding {
            Encoding::Basis => DeviceInput::I64(ptr as *const usize),
            _ => DeviceInput::F64(ptr as *const f64),
        }
    }

    /// Encode one sample already on the device (`f64`, or `int64` for basis).
    ///
    /// # Safety
    /// `input_d` must hold `input_len` elements on this engine's device.
    pub unsafe fn encode_from_gpu_ptr(
        &self,
        input_d: *const c_void,
        input_len: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        // SAFETY: forwarded from the caller's contract.
        unsafe {
            self.encode_from_gpu_ptr_with_stream(
                input_d,
                input_len,
                num_qubits,
                encoding_method,
                std::ptr::null_mut(),
            )
        }
    }

    /// # Safety
    /// As [`QdpEngine::encode_from_gpu_ptr`]; `stream` must be live.
    pub unsafe fn encode_from_gpu_ptr_with_stream(
        &self,
        input_d: *const c_void,
        input_len: usize,
        num_qubits: usize,
        encoding_method: &str,
        stream: *mut c_void,
    ) -> Result<*mut DLManagedTensor> {
        let encoding = Encoding::from_str_ci(encoding_method)?;
        self.encode(
            Input::Device {
                ptr: Self::device_ptr_for(input_d, encoding),
                stream,
            },
            Shape::new(1, input_len),
            num_qubits,
            encoding,
        )
    }

    /// Encode a batch already on the device (`f64`, or `int64` for basis).
    ///
    /// # Safety
    /// `input_batch_d` must hold `num_samples * sample_size` elements on this
    /// engine's device.
    pub unsafe fn encode_batch_from_gpu_ptr(
        &self,
        input_batch_d: *const c_void,
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        // SAFETY: forwarded from the caller's contract.
        unsafe {
            self.encode_batch_from_gpu_ptr_with_stream(
                input_batch_d,
                num_samples,
                sample_size,
                num_qubits,
                encoding_method,
                std::ptr::null_mut(),
            )
        }
    }

    /// # Safety
    /// As [`QdpEngine::encode_batch_from_gpu_ptr`]; `stream` must be live.
    pub unsafe fn encode_batch_from_gpu_ptr_with_stream(
        &self,
        input_batch_d: *const c_void,
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
        stream: *mut c_void,
    ) -> Result<*mut DLManagedTensor> {
        let encoding = Encoding::from_str_ci(encoding_method)?;
        self.encode(
            Input::Device {
                ptr: Self::device_ptr_for(input_batch_d, encoding),
                stream,
            },
            Shape::new(num_samples, sample_size),
            num_qubits,
            encoding,
        )
    }
}
