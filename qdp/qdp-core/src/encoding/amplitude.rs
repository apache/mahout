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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MahoutError;
    use cudarc::driver::DeviceSlice;

    #[test]
    fn reject_sample_size_zero() {
        let enc = AmplitudeEncoder;
        match enc.validate_sample_size(0) {
            Err(MahoutError::InvalidInput(msg)) => assert!(msg.contains("zero")),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[test]
    fn reject_sample_size_exceeds_stage() {
        let enc = AmplitudeEncoder;
        match enc.validate_sample_size(STAGE_SIZE_ELEMENTS + 1) {
            Err(MahoutError::InvalidInput(msg)) => assert!(msg.contains("exceeds")),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[test]
    fn accept_valid_sample_size() {
        let enc = AmplitudeEncoder;
        assert!(enc.validate_sample_size(4).is_ok());
    }

    #[test]
    fn accept_max_valid_sample_size() {
        let enc = AmplitudeEncoder;
        assert!(enc.validate_sample_size(STAGE_SIZE_ELEMENTS).is_ok());
    }

    #[test]
    fn needs_staging_copy_returns_true() {
        assert!(AmplitudeEncoder.needs_staging_copy());
    }

    #[test]
    fn init_state_allocates_norm_buffer() {
        let engine = match QdpEngine::new(0) {
            Ok(e) => e,
            Err(err) => {
                eprintln!(
                    "skipping init_state_allocates_norm_buffer: failed to create QdpEngine: {}",
                    err
                );
                return;
            }
        };
        let enc = AmplitudeEncoder;
        let sample_size = 4;
        let num_qubits = 2; // the third param is num_qubits, not batch_size
        let state = enc
            .init_state(&engine, sample_size, num_qubits)
            .expect("init_state should succeed with valid params");

        let expected_len = super::STAGE_SIZE_ELEMENTS / sample_size;
        assert_eq!(
            state.norm_buffer.len(),
            expected_len,
            "norm_buffer length should match max_samples_in_chunk"
        );
    }

    #[test]
    fn test_stream_encode_end_to_end() {
        let engine = match QdpEngine::new(0) {
            Ok(e) => e,
            Err(err) => {
                eprintln!(
                    "skipping test_stream_encode_end_to_end: failed to create QdpEngine: {}",
                    err
                );
                return;
            }
        };

        let temp_path = std::env::temp_dir().join("test_amplitude_stream_encode.parquet");
        let temp_path_str = temp_path.to_str().unwrap();

        let data: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        crate::io::write_parquet(&temp_path, &data, None).unwrap();

        let num_qubits = 2; // sample_size = 4
        let result =
            crate::encoding::stream_encode(&engine, temp_path_str, num_qubits, AmplitudeEncoder);

        assert!(
            result.is_ok(),
            "stream_encode should succeed with valid data"
        );

        // Clean up unmanaged pointers generated by DLPack conversion
        unsafe {
            let tensor_ptr = result.unwrap();
            if !tensor_ptr.is_null() {
                let tensor = &*tensor_ptr;
                if let Some(deleter) = tensor.deleter {
                    deleter(tensor_ptr);
                }
            }
        }
        std::fs::remove_file(&temp_path).unwrap_or(());
    }
}
