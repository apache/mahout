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

//! Basis encoding implementation.

use std::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr};
use qdp_kernels::launch_basis_encode_batch;

use super::{ChunkEncoder, STAGE_SIZE_ELEMENTS};
use crate::gpu::PipelineContext;
use crate::gpu::memory::PinnedHostBuffer;
use crate::{MahoutError, QdpEngine, Result};

/// Basis encoder state containing reusable buffers.
pub(crate) struct BasisEncoderState {
    /// Reusable CPU buffer for validated indices.
    indices_cpu: Vec<usize>,
    /// Reusable GPU buffer for indices.
    indices_gpu: CudaSlice<usize>,
}

/// Basis encoding: maps integer indices to computational basis states.
pub(crate) struct BasisEncoder;

impl ChunkEncoder for BasisEncoder {
    type State = BasisEncoderState;

    fn needs_staging_copy(&self) -> bool {
        // Basis encoding validates indices on CPU and uploads directly,
        // so we don't need the staging buffer H2D copy.
        false
    }

    fn validate_sample_size(&self, sample_size: usize) -> Result<()> {
        if sample_size != 1 {
            return Err(MahoutError::InvalidInput(format!(
                "Basis encoding requires sample_size=1 (one index per sample), got {}",
                sample_size
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
        // For basis encoding, sample_size is always 1, so max samples = STAGE_SIZE_ELEMENTS
        let max_samples_in_chunk = STAGE_SIZE_ELEMENTS / sample_size;

        // Pre-allocate CPU buffer for indices
        let indices_cpu = Vec::with_capacity(max_samples_in_chunk);

        // Pre-allocate GPU buffer for indices
        let indices_gpu =
            unsafe { engine.device.alloc::<usize>(max_samples_in_chunk) }.map_err(|e| {
                MahoutError::MemoryAllocation(format!(
                    "Failed to allocate GPU indices buffer: {:?}",
                    e
                ))
            })?;

        Ok(BasisEncoderState {
            indices_cpu,
            indices_gpu,
        })
    }

    fn encode_chunk(
        &self,
        state: &mut Self::State,
        engine: &QdpEngine,
        ctx: &PipelineContext,
        host_buffer: &PinnedHostBuffer,
        _dev_ptr: u64,
        samples_in_chunk: usize,
        _sample_size: usize,
        state_ptr_offset: *mut c_void,
        state_len: usize,
        num_qubits: usize,
        global_sample_offset: usize,
    ) -> Result<()> {
        unsafe {
            crate::profile_scope!("GPU::BatchEncode");

            // Clear and reuse CPU buffer for validated indices
            state.indices_cpu.clear();

            // Validate and convert indices on CPU
            let data_slice = std::slice::from_raw_parts(host_buffer.ptr(), samples_in_chunk);
            for (i, &val) in data_slice.iter().enumerate() {
                if !val.is_finite() {
                    return Err(MahoutError::InvalidInput(format!(
                        "Sample {}: basis index must be finite",
                        global_sample_offset + i
                    )));
                }
                if val < 0.0 {
                    return Err(MahoutError::InvalidInput(format!(
                        "Sample {}: basis index must be non-negative",
                        global_sample_offset + i
                    )));
                }
                if val.fract() != 0.0 {
                    return Err(MahoutError::InvalidInput(format!(
                        "Sample {}: basis index must be an integer, got {}",
                        global_sample_offset + i,
                        val
                    )));
                }
                let index = val as usize;
                if index >= state_len {
                    return Err(MahoutError::InvalidInput(format!(
                        "Sample {}: basis index {} exceeds state size {} (max: {})",
                        global_sample_offset + i,
                        index,
                        state_len,
                        state_len - 1
                    )));
                }
                state.indices_cpu.push(index);
            }

            // Copy indices to pre-allocated GPU buffer (slice to match actual chunk size)
            let mut gpu_slice = state.indices_gpu.slice_mut(0..samples_in_chunk);
            engine
                .device
                .htod_sync_copy_into(&state.indices_cpu, &mut gpu_slice)
                .map_err(|e| {
                    MahoutError::MemoryAllocation(format!(
                        "Failed to upload basis indices to GPU: {:?}",
                        e
                    ))
                })?;

            // Launch basis encoding kernel
            {
                crate::profile_scope!("GPU::BasisEncodeBatch");
                let ret = launch_basis_encode_batch(
                    *state.indices_gpu.device_ptr() as *const usize,
                    state_ptr_offset,
                    samples_in_chunk,
                    state_len,
                    num_qubits as u32,
                    ctx.stream_compute.stream as *mut c_void,
                );
                if ret != 0 {
                    return Err(MahoutError::KernelLaunch(format!(
                        "Basis encode kernel error: {}",
                        ret
                    )));
                }
            }
        }
        Ok(())
    }
}
