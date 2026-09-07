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

// Allow unused_unsafe: CUDA FFI and kernel functions are unsafe in CUDA builds but safe stubs in no-CUDA builds.
// The compiler can't statically determine which path is taken.
#![allow(unused_unsafe)]

mod compat;
pub mod dlpack;
#[cfg(target_os = "linux")]
mod encoding;
pub mod error;
pub mod estimate;
pub mod gpu;
pub mod io;
mod platform;
pub mod preprocessing;
pub mod reader;
pub mod readers;
#[cfg(feature = "remote-io")]
pub mod remote;
pub mod tf_proto;
pub mod types;
#[macro_use]
mod profiling;

pub use error::{MahoutError, Result, cuda_error_to_string};
pub use estimate::{MemoryEstimate, estimate_memory};
pub use gpu::cuda_ffi::cuda_runtime_available;
pub use gpu::memory::Precision;
pub use reader::{FloatElem, NullHandling, handle_float32_nulls, handle_float64_nulls};
pub use types::{Dtype, Encoding};

// Throughput/latency pipeline runner: single path using QdpEngine and encode_batch in Rust.
#[cfg(target_os = "linux")]
mod pipeline_runner;

#[cfg(target_os = "linux")]
pub use pipeline_runner::{
    PipelineConfig, PipelineIterator, PipelineRunResult, run_latency_pipeline,
    run_throughput_pipeline,
};

use std::sync::Arc;

use crate::dlpack::DLManagedTensor;
use cudarc::driver::CudaDevice;

pub use gpu::kernels::{DeviceDtype, DeviceInput, HostInput, Input, Kernel, Shape};

/// GPU encoding engine: owns a device and the output precision.
#[derive(Clone)]
pub struct QdpEngine {
    device: Arc<CudaDevice>,
    precision: Precision,
}

impl QdpEngine {
    /// Initialise on `device_id` with float32 output.
    pub fn new(device_id: usize) -> Result<Self> {
        Self::new_with_precision(device_id, Precision::Float32)
    }

    /// Initialise on `device_id` with the given output precision.
    pub fn new_with_precision(device_id: usize, precision: Precision) -> Result<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            MahoutError::Cuda(format!(
                "Failed to initialize CUDA device {}: {:?}",
                device_id, e
            ))
        })?;
        Ok(Self { device, precision })
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    #[cfg(target_os = "linux")]
    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e)))
    }

    /// Encode a batch and hand it back as a DLPack tensor of shape
    /// `[num_samples, 2^num_qubits]` in the engine's precision.
    ///
    /// This is the one encode entry point. `input` may live on the host or
    /// already on this device, as `f32` or `f64` (or `int64` basis indices);
    /// a single sample is a batch of one.
    ///
    /// For [`Input::Device`] the caller is responsible for the pointer
    /// holding `shape.num_samples * shape.sample_size` elements on this
    /// engine's device, valid on the given stream until the call returns.
    /// The pointer is checked to be device memory on this device before use.
    pub fn encode(
        &self,
        input: Input,
        shape: Shape,
        num_qubits: usize,
        encoding: Encoding,
    ) -> Result<*mut DLManagedTensor> {
        let state = self.encode_state(input, shape, num_qubits, encoding)?;
        let dlpack_ptr = {
            crate::profile_scope!("DLPack::Wrap");
            state.to_dlpack()
        };
        Ok(dlpack_ptr)
    }

    /// [`QdpEngine::encode`] without the DLPack wrapping.
    pub fn encode_state(
        &self,
        input: Input,
        shape: Shape,
        num_qubits: usize,
        encoding: Encoding,
    ) -> Result<gpu::GpuStateVector> {
        crate::profile_scope!("Mahout::Encode");
        // A float64 engine promises float64 accuracy: widen f32 host input on
        // the CPU rather than encode it with the f32 kernel and up-convert.
        // (Data already on the device is encoded at its own precision.)
        let widened: Vec<f64>;
        let input = match (input, self.precision) {
            (Input::Host(HostInput::F32(s)), Precision::Float64) => {
                widened = s.iter().map(|&v| v as f64).collect();
                Input::Host(HostInput::F64(&widened))
            }
            (other, _) => other,
        };
        let state =
            gpu::kernels::encode(&self.device, encoding.encoder(), input, shape, num_qubits)?;
        state.to_precision(&self.device, self.precision)
    }

    /// Stream a Parquet file through the GPU, one chunk at a time.
    pub fn encode_from_parquet(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        platform::encode_from_parquet(self, path, num_qubits, encoding_method)
    }

    pub fn encode_from_arrow_ipc(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromArrowIPC");
        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadArrowIPCBatch");
            crate::io::read_arrow_ipc_batch(path)?
        };
        self.encode_host_f64(
            &batch_data,
            num_samples,
            sample_size,
            num_qubits,
            encoding_method,
        )
    }

    pub fn encode_from_numpy(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromNumpy");
        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadNumpyBatch");
            crate::io::read_numpy_batch(path)?
        };
        self.encode_host_f64(
            &batch_data,
            num_samples,
            sample_size,
            num_qubits,
            encoding_method,
        )
    }

    pub fn encode_from_torch(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromTorch");
        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadTorchBatch");
            crate::io::read_torch_batch(path)?
        };
        self.encode_host_f64(
            &batch_data,
            num_samples,
            sample_size,
            num_qubits,
            encoding_method,
        )
    }

    pub fn encode_from_tensorflow(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromTensorFlow");
        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadTensorFlowBatch");
            crate::io::read_tensorflow_batch(path)?
        };
        self.encode_host_f64(
            &batch_data,
            num_samples,
            sample_size,
            num_qubits,
            encoding_method,
        )
    }

    fn encode_host_f64(
        &self,
        data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        self.encode(
            Input::Host(HostInput::F64(data)),
            Shape::new(num_samples, sample_size),
            num_qubits,
            Encoding::from_str_ci(encoding_method)?,
        )
    }
}
