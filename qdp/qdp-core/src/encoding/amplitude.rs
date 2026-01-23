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

//! Amplitude encoding implementation.

// Allow unused_unsafe: qdp_kernels functions are unsafe in CUDA builds but safe stubs in no-CUDA builds.
// The compiler can't statically determine which path is taken.
#![allow(unused_unsafe)]

use std::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtrMut};
use qdp_kernels::{launch_amplitude_encode_batch, launch_l2_norm_batch};

use super::{ChunkEncoder, STAGE_SIZE_ELEMENTS};
use crate::gpu::PipelineContext;
use crate::gpu::memory::PinnedHostBuffer;
use crate::{MahoutError, QdpEngine, Result};

/// Amplitude encoder state containing the norm buffer.
pub(crate) struct AmplitudeEncoderState {
    norm_buffer: CudaSlice<f64>,
}

/// Amplitude encoding: maps classical vectors to quantum state amplitudes.
pub(crate) struct AmplitudeEncoder;

impl ChunkEncoder for AmplitudeEncoder {
    type State = AmplitudeEncoderState;

    fn validate_sample_size(&self, sample_size: usize) -> Result<()> {
        if sample_size == 0 {
            return Err(MahoutError::InvalidInput(
                "Sample size cannot be zero".into(),
            ));
        }
        if sample_size > STAGE_SIZE_ELEMENTS {
            return Err(MahoutError::InvalidInput(format!(
                "Sample size {} exceeds staging buffer capacity {}",
                sample_size, STAGE_SIZE_ELEMENTS
            )));
        }
        Ok(())
    }

    fn init_state(
        &self,
        engine: &QdpEngine,
        sample_size: usize,
        _num_qubits: usize,
    ) -> Result<Self::State> {
        let max_samples_in_chunk = STAGE_SIZE_ELEMENTS / sample_size;
        let norm_buffer = engine
            .device
            .alloc_zeros::<f64>(max_samples_in_chunk)
            .map_err(|e| {
                MahoutError::MemoryAllocation(format!("Failed to allocate norm buffer: {:?}", e))
            })?;
        Ok(AmplitudeEncoderState { norm_buffer })
    }

    fn encode_chunk(
        &self,
        state: &mut Self::State,
        _engine: &QdpEngine,
        ctx: &PipelineContext,
        _host_buffer: &PinnedHostBuffer,
        dev_ptr: u64,
        samples_in_chunk: usize,
        sample_size: usize,
        state_ptr_offset: *mut c_void,
        state_len: usize,
        _num_qubits: usize,
        _global_sample_offset: usize,
    ) -> Result<()> {
        unsafe {
            crate::profile_scope!("GPU::BatchEncode");

            // Compute L2 norms
            {
                crate::profile_scope!("GPU::NormBatch");
                let ret = launch_l2_norm_batch(
                    dev_ptr as *const f64,
                    samples_in_chunk,
                    sample_size,
                    *state.norm_buffer.device_ptr_mut() as *mut f64,
                    ctx.stream_compute.stream as *mut c_void,
                );
                if ret != 0 {
                    return Err(MahoutError::KernelLaunch(format!(
                        "Norm kernel error: {}",
                        ret
                    )));
                }
            }

            // Encode amplitudes
            {
                crate::profile_scope!("GPU::EncodeBatch");
                let ret = launch_amplitude_encode_batch(
                    dev_ptr as *const f64,
                    state_ptr_offset,
                    *state.norm_buffer.device_ptr_mut() as *const f64,
                    samples_in_chunk,
                    sample_size,
                    state_len,
                    ctx.stream_compute.stream as *mut c_void,
                );
                if ret != 0 {
                    return Err(MahoutError::KernelLaunch(format!(
                        "Encode kernel error: {}",
                        ret
                    )));
                }
            }
        }
        Ok(())
    }
}
