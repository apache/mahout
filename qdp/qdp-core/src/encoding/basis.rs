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

use cudarc::driver::DevicePtr;
use qdp_kernels::launch_basis_encode_batch;

use super::ChunkEncoder;
use crate::gpu::PipelineContext;
use crate::gpu::memory::PinnedHostBuffer;
use crate::{MahoutError, QdpEngine, Result};

/// Basis encoder has no persistent state.
pub(crate) struct BasisEncoderState;

/// Basis encoding: maps integer indices to computational basis states.
pub(crate) struct BasisEncoder;

impl ChunkEncoder for BasisEncoder {
    type State = BasisEncoderState;

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
        _engine: &QdpEngine,
        _sample_size: usize,
        _num_qubits: usize,
    ) -> Result<Self::State> {
        Ok(BasisEncoderState)
    }

    fn encode_chunk(
        &self,
        _state: &mut Self::State,
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

            // Validate and convert indices on CPU
            let indices_cpu: Vec<usize> = {
                let data_slice = std::slice::from_raw_parts(host_buffer.ptr(), samples_in_chunk);
                data_slice
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| {
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
                        Ok(index)
                    })
                    .collect::<Result<Vec<_>>>()?
            };

            // Upload indices to GPU
            let indices_gpu = engine.device.htod_sync_copy(&indices_cpu).map_err(|e| {
                MahoutError::MemoryAllocation(format!(
                    "Failed to upload basis indices to GPU: {:?}",
                    e
                ))
            })?;

            // Launch basis encoding kernel
            {
                crate::profile_scope!("GPU::BasisEncodeBatch");
                let ret = launch_basis_encode_batch(
                    *indices_gpu.device_ptr() as *const usize,
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
