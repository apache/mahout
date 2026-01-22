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

//! Angle encoding implementation.

// Allow unused_unsafe: qdp_kernels functions are unsafe in CUDA builds but safe stubs in no-CUDA builds.
// The compiler can't statically determine which path is taken.
#![allow(unused_unsafe)]

use std::ffi::c_void;

use qdp_kernels::launch_angle_encode_batch;

use super::{ChunkEncoder, STAGE_SIZE_ELEMENTS};
use crate::gpu::PipelineContext;
use crate::gpu::memory::PinnedHostBuffer;
use crate::{MahoutError, QdpEngine, Result};

/// Angle encoder state (no persistent buffers required).
pub(crate) struct AngleEncoderState;

/// Angle encoding: maps per-qubit angles to product state amplitudes.
pub(crate) struct AngleEncoder;

impl ChunkEncoder for AngleEncoder {
    type State = AngleEncoderState;

    fn validate_sample_size(&self, sample_size: usize) -> Result<()> {
        if sample_size == 0 {
            return Err(MahoutError::InvalidInput(
                "Angle encoding requires sample_size > 0".into(),
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
        _engine: &QdpEngine,
        sample_size: usize,
        num_qubits: usize,
    ) -> Result<Self::State> {
        if num_qubits == 0 || num_qubits > 30 {
            return Err(MahoutError::InvalidInput(format!(
                "Number of qubits {} must be between 1 and 30",
                num_qubits
            )));
        }
        if sample_size != num_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "Angle encoding expects sample_size={} (one angle per qubit), got {}",
                num_qubits, sample_size
            )));
        }
        Ok(AngleEncoderState)
    }

    fn encode_chunk(
        &self,
        _state: &mut Self::State,
        _engine: &QdpEngine,
        ctx: &PipelineContext,
        host_buffer: &PinnedHostBuffer,
        dev_ptr: u64,
        samples_in_chunk: usize,
        sample_size: usize,
        state_ptr_offset: *mut c_void,
        state_len: usize,
        num_qubits: usize,
        global_sample_offset: usize,
    ) -> Result<()> {
        let total_values = samples_in_chunk.checked_mul(sample_size).ok_or_else(|| {
            MahoutError::MemoryAllocation(format!(
                "Angle chunk size overflow: {} * {}",
                samples_in_chunk, sample_size
            ))
        })?;

        let data_slice = unsafe { std::slice::from_raw_parts(host_buffer.ptr(), total_values) };
        for (i, &val) in data_slice.iter().enumerate() {
            if !val.is_finite() {
                let sample_idx = global_sample_offset + (i / sample_size);
                let angle_idx = i % sample_size;
                return Err(MahoutError::InvalidInput(format!(
                    "Sample {} angle {} must be finite, got {}",
                    sample_idx, angle_idx, val
                )));
            }
        }

        crate::profile_scope!("GPU::BatchEncode");
        let ret = unsafe {
            launch_angle_encode_batch(
                dev_ptr as *const f64,
                state_ptr_offset,
                samples_in_chunk,
                state_len,
                num_qubits as u32,
                ctx.stream_compute.stream as *mut c_void,
            )
        };
        if ret != 0 {
            return Err(MahoutError::KernelLaunch(format!(
                "Angle encode kernel error: {}",
                ret
            )));
        }
        Ok(())
    }
}
