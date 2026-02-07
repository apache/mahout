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

//! Streaming encoding implementations for different quantum encoding methods.

mod amplitude;
mod angle;
mod basis;

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use cudarc::driver::{CudaDevice, DevicePtr};

/// Guard that ensures GPU synchronization and IO thread cleanup on drop.
/// Used to handle early returns in `stream_encode`.
struct CleanupGuard<'a> {
    device: &'a Arc<CudaDevice>,
    io_handle: Option<JoinHandle<()>>,
}

impl<'a> CleanupGuard<'a> {
    fn new(device: &'a Arc<CudaDevice>, io_handle: JoinHandle<()>) -> Self {
        Self {
            device,
            io_handle: Some(io_handle),
        }
    }

    /// Defuse the guard and return the IO handle for explicit cleanup.
    /// After calling this, drop() will not perform cleanup.
    fn defuse(mut self) -> JoinHandle<()> {
        self.io_handle.take().expect("IO handle already taken")
    }
}

impl Drop for CleanupGuard<'_> {
    fn drop(&mut self) {
        // Best-effort cleanup on early return
        let _ = self.device.synchronize();
        if let Some(handle) = self.io_handle.take() {
            let _ = handle.join();
        }
    }
}

use crate::dlpack::DLManagedTensor;
use crate::gpu::PipelineContext;
use crate::gpu::memory::{GpuStateVector, PinnedHostBuffer};
use crate::reader::StreamingDataReader;
use crate::{MahoutError, QdpEngine, Result};

/// 512MB staging buffer for large Parquet row groups (reduces fragmentation)
pub(crate) const STAGE_SIZE_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const STAGE_SIZE_ELEMENTS: usize = STAGE_SIZE_BYTES / std::mem::size_of::<f64>();

pub(crate) type FullBufferResult = std::result::Result<(PinnedHostBuffer, usize), MahoutError>;
pub(crate) type FullBufferChannel = (SyncSender<FullBufferResult>, Receiver<FullBufferResult>);

/// Trait for chunk-based quantum state encoding.
///
/// Implementations provide the encoding-specific logic while the shared
/// streaming pipeline handles IO, buffering, and GPU memory management.
pub(crate) trait ChunkEncoder {
    /// Encoder-specific state (e.g., norm buffer for amplitude encoding).
    type State;

    /// Validate that the sample size is appropriate for this encoding method.
    fn validate_sample_size(&self, sample_size: usize) -> Result<()>;

    /// Whether this encoder needs the staging buffer H2D copy.
    ///
    /// If false, the streaming pipeline will skip the async copy to device
    /// staging buffer, avoiding unnecessary memory bandwidth overhead.
    /// Encoders that process data on CPU before uploading should return false.
    fn needs_staging_copy(&self) -> bool {
        true
    }

    /// Initialize encoder-specific state.
    fn init_state(
        &self,
        engine: &QdpEngine,
        sample_size: usize,
        num_qubits: usize,
    ) -> Result<Self::State>;

    /// Encode a chunk of samples to quantum states.
    ///
    /// # Arguments
    /// * `state` - Encoder-specific state
    /// * `engine` - QDP engine for GPU operations
    /// * `ctx` - Pipeline context for async operations
    /// * `host_buffer` - Pinned host buffer containing input data
    /// * `dev_ptr` - Device pointer to staging buffer with copied data
    /// * `samples_in_chunk` - Number of samples in this chunk
    /// * `sample_size` - Size of each sample in f64 elements
    /// * `state_ptr_offset` - Pointer to output location in state vector
    /// * `state_len` - Length of each quantum state (2^num_qubits)
    /// * `num_qubits` - Number of qubits
    #[allow(clippy::too_many_arguments)]
    fn encode_chunk(
        &self,
        state: &mut Self::State,
        engine: &QdpEngine,
        ctx: &PipelineContext,
        host_buffer: &PinnedHostBuffer,
        dev_ptr: u64,
        samples_in_chunk: usize,
        sample_size: usize,
        state_ptr_offset: *mut c_void,
        state_len: usize,
        num_qubits: usize,
        global_sample_offset: usize,
    ) -> Result<()>;
}

/// Shared streaming pipeline for encoding data from Parquet files.
///
/// This function handles all the common IO, buffering, and GPU memory
/// management logic. The actual encoding is delegated to the `ChunkEncoder`.
pub(crate) fn stream_encode<E: ChunkEncoder>(
    engine: &QdpEngine,
    path: &str,
    num_qubits: usize,
    encoder: E,
) -> Result<*mut DLManagedTensor> {
    // Initialize reader
    let mut reader_core = crate::io::ParquetBlockReader::new(path, None)?;
    let num_samples = reader_core.total_rows;

    // Allocate output state vector
    let total_state_vector =
        GpuStateVector::new_batch(&engine.device, num_samples, num_qubits, engine.precision())?;
    const PIPELINE_EVENT_SLOTS: usize = 2;
    let ctx = PipelineContext::new(&engine.device, PIPELINE_EVENT_SLOTS)?;

    // Check if encoder needs staging buffers before allocating
    let needs_staging_copy = encoder.needs_staging_copy();

    // Double-buffered device staging (only allocated if needed)
    let dev_staging = if needs_staging_copy {
        let dev_in_a = unsafe { engine.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
            .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
        let dev_in_b = unsafe { engine.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
            .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
        Some((dev_in_a, dev_in_b))
    } else {
        None
    };

    // Channel setup for async IO
    let (full_buf_tx, full_buf_rx): FullBufferChannel = sync_channel(2);
    let (empty_buf_tx, empty_buf_rx): (SyncSender<PinnedHostBuffer>, _) = sync_channel(2);

    // Read first chunk to determine sample size
    let mut host_buf_first = PinnedHostBuffer::new(STAGE_SIZE_ELEMENTS)?;
    let first_len = reader_core.read_chunk(host_buf_first.as_slice_mut())?;

    let sample_size = reader_core
        .get_sample_size()
        .ok_or_else(|| MahoutError::InvalidInput("Could not determine sample size".into()))?;

    // Validate sample size for this encoder
    encoder.validate_sample_size(sample_size)?;

    // Initialize encoder-specific state
    let mut encoder_state = encoder.init_state(engine, sample_size, num_qubits)?;

    let state_len = 1 << num_qubits;

    // Send first buffer to processing
    full_buf_tx
        .send(Ok((host_buf_first, first_len)))
        .map_err(|_| MahoutError::Io("Failed to send first buffer".into()))?;

    // Send second empty buffer for IO thread
    empty_buf_tx
        .send(PinnedHostBuffer::new(STAGE_SIZE_ELEMENTS)?)
        .map_err(|_| MahoutError::Io("Failed to send second buffer".into()))?;

    // Spawn IO thread
    let mut reader = reader_core;
    let io_handle = thread::spawn(move || {
        loop {
            let mut buffer = match empty_buf_rx.recv() {
                Ok(b) => b,
                Err(_) => break,
            };

            let result = reader
                .read_chunk(buffer.as_slice_mut())
                .map(|len| (buffer, len));

            let should_break = match &result {
                Ok((_, len)) => *len == 0,
                Err(_) => true,
            };

            if full_buf_tx.send(result).is_err() {
                break;
            }

            if should_break {
                break;
            }
        }
    });

    // Create cleanup guard to ensure resources are released on early return
    let cleanup_guard = CleanupGuard::new(&engine.device, io_handle);

    // Main processing loop
    let mut global_sample_offset: usize = 0;
    let mut use_dev_a = true;

    loop {
        let (host_buffer, current_len) = match full_buf_rx.recv() {
            Ok(Ok((buffer, len))) => (buffer, len),
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(MahoutError::Io("IO thread disconnected".into())),
        };

        if current_len == 0 {
            break;
        }

        if current_len % sample_size != 0 {
            return Err(MahoutError::InvalidInput(format!(
                "Chunk length {} is not a multiple of sample size {}",
                current_len, sample_size
            )));
        }

        let samples_in_chunk = current_len / sample_size;
        if samples_in_chunk > 0 {
            let event_slot = if use_dev_a { 0 } else { 1 };
            // Get device pointer from staging buffers (0 if not allocated)
            let dev_ptr = dev_staging
                .as_ref()
                .map(|(a, b)| {
                    if use_dev_a {
                        *a.device_ptr()
                    } else {
                        *b.device_ptr()
                    }
                })
                .unwrap_or(0);

            unsafe {
                crate::profile_scope!("GPU::Dispatch");

                // Async copy to device (only if staging buffers are allocated)
                if dev_staging.is_some() {
                    ctx.async_copy_to_device(
                        host_buffer.ptr() as *const c_void,
                        dev_ptr as *mut c_void,
                        current_len,
                    )?;
                    ctx.record_copy_done(event_slot)?;
                    ctx.wait_for_copy(event_slot)?;
                }

                // Calculate output offset
                let offset_elements =
                    global_sample_offset.checked_mul(state_len).ok_or_else(|| {
                        MahoutError::MemoryAllocation(format!(
                            "Offset calculation overflow: {} * {}",
                            global_sample_offset, state_len
                        ))
                    })?;

                let offset_bytes = offset_elements
                    .checked_mul(std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
                    .ok_or_else(|| {
                        MahoutError::MemoryAllocation(format!(
                            "Offset bytes calculation overflow: {} * {}",
                            offset_elements,
                            std::mem::size_of::<qdp_kernels::CuDoubleComplex>()
                        ))
                    })?;

                let state_ptr_offset = total_state_vector
                    .ptr_void()
                    .cast::<u8>()
                    .add(offset_bytes)
                    .cast::<c_void>();

                // Delegate to encoder
                encoder.encode_chunk(
                    &mut encoder_state,
                    engine,
                    &ctx,
                    &host_buffer,
                    dev_ptr,
                    samples_in_chunk,
                    sample_size,
                    state_ptr_offset,
                    state_len,
                    num_qubits,
                    global_sample_offset,
                )?;

                if dev_staging.is_some() {
                    ctx.sync_copy_stream()?;
                }
            }

            global_sample_offset = global_sample_offset
                .checked_add(samples_in_chunk)
                .ok_or_else(|| {
                    MahoutError::MemoryAllocation(format!(
                        "Sample offset overflow: {} + {}",
                        global_sample_offset, samples_in_chunk
                    ))
                })?;
            use_dev_a = !use_dev_a;
        }

        let _ = empty_buf_tx.send(host_buffer);
    }

    // Defuse guard for explicit cleanup with proper error handling
    let io_handle = cleanup_guard.defuse();

    engine
        .device
        .synchronize()
        .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
    io_handle
        .join()
        .map_err(|e| MahoutError::Io(format!("IO thread panicked: {:?}", e)))?;

    let dlpack_ptr = total_state_vector.to_dlpack();
    Ok(dlpack_ptr)
}

/// Encode data from a Parquet file using the specified encoding method.
pub(crate) fn encode_from_parquet(
    engine: &QdpEngine,
    path: &str,
    num_qubits: usize,
    encoding_method: &str,
) -> Result<*mut DLManagedTensor> {
    match encoding_method {
        "amplitude" => {
            crate::profile_scope!("Mahout::EncodeAmplitudeFromParquet");
            stream_encode(engine, path, num_qubits, amplitude::AmplitudeEncoder)
        }
        "angle" => {
            crate::profile_scope!("Mahout::EncodeAngleFromParquet");
            stream_encode(engine, path, num_qubits, angle::AngleEncoder)
        }
        "basis" => {
            crate::profile_scope!("Mahout::EncodeBasisFromParquet");
            stream_encode(engine, path, num_qubits, basis::BasisEncoder)
        }
        _ => Err(MahoutError::NotImplemented(format!(
            "Encoding method '{}' not supported for streaming",
            encoding_method
        ))),
    }
}
