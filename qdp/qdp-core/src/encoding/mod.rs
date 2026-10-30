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

/// Close the main thread's ends of the IO channels, then wait for the IO
/// thread to finish.
///
/// Both endpoints must be dropped *before* `join()`. The IO thread is either
/// parked in `empty_buf_rx.recv()`, which only returns once the matching
/// `empty_buf_tx` is dropped, or blocked in `full_buf_tx.send()` on a bounded
/// channel nobody is draining any more, which only fails once `full_buf_rx` is
/// dropped. Joining while either endpoint is still alive therefore hangs
/// forever on any early exit from the encoding loop.
///
/// The payload types are irrelevant here, since the endpoints are only
/// dropped, so this is generic over them; that also lets the ordering be
/// unit tested without a GPU.
fn shutdown_io_thread<T, U>(
    empty_buf_tx: Option<SyncSender<T>>,
    full_buf_rx: Option<Receiver<U>>,
    io_handle: JoinHandle<()>,
) -> thread::Result<()> {
    drop(empty_buf_tx);
    drop(full_buf_rx);
    io_handle.join()
}

/// Guard that ensures GPU synchronization and IO thread cleanup on drop.
/// Used to handle early returns in `stream_encode`.
///
/// The guard owns the main thread's ends of the IO channels, so they are
/// always closed before the IO thread is joined, no matter which exit path
/// (early return, `?`, or unwinding) drops the guard.
struct CleanupGuard<'a> {
    device: &'a Arc<CudaDevice>,
    io_handle: Option<JoinHandle<()>>,
    empty_buf_tx: Option<SyncSender<PinnedHostBuffer>>,
    full_buf_rx: Option<Receiver<FullBufferResult>>,
}

impl<'a> CleanupGuard<'a> {
    fn new(
        device: &'a Arc<CudaDevice>,
        io_handle: JoinHandle<()>,
        empty_buf_tx: SyncSender<PinnedHostBuffer>,
        full_buf_rx: Receiver<FullBufferResult>,
    ) -> Self {
        Self {
            device,
            io_handle: Some(io_handle),
            empty_buf_tx: Some(empty_buf_tx),
            full_buf_rx: Some(full_buf_rx),
        }
    }

    /// Sender used to recycle drained buffers back to the IO thread.
    fn empty_buf_tx(&self) -> &SyncSender<PinnedHostBuffer> {
        self.empty_buf_tx
            .as_ref()
            .expect("empty buffer sender already taken")
    }

    /// Receiver for buffers filled by the IO thread.
    fn full_buf_rx(&self) -> &Receiver<FullBufferResult> {
        self.full_buf_rx
            .as_ref()
            .expect("full buffer receiver already taken")
    }

    /// Defuse the guard and return the IO handle for explicit cleanup.
    /// After calling this, drop() will not join the IO thread; it still closes
    /// the channel endpoints it owns, so the returned handle can be joined
    /// without deadlocking.
    ///
    /// Note that `Drop` also runs `device.synchronize()` and discards its
    /// error, so that synchronization happens before any checked
    /// `synchronize()` the caller performs afterwards.
    fn defuse(mut self) -> JoinHandle<()> {
        self.io_handle.take().expect("IO handle already taken")
    }
}

impl Drop for CleanupGuard<'_> {
    fn drop(&mut self) {
        // Best-effort cleanup on early return
        let _ = self.device.synchronize();
        // Taken unconditionally: even when the guard has been defused, the
        // channel endpoints must be closed so the IO thread can observe the
        // shutdown and the caller's explicit join() can complete.
        let empty_buf_tx = self.empty_buf_tx.take();
        let full_buf_rx = self.full_buf_rx.take();
        if let Some(handle) = self.io_handle.take() {
            let _ = shutdown_io_thread(empty_buf_tx, full_buf_rx, handle);
        }
    }
}

use crate::dlpack::DLManagedTensor;
use crate::gpu::PipelineContext;
use crate::gpu::memory::{GpuStateVector, PinnedHostBuffer};
use crate::reader::StreamingDataReader;
use crate::types::Encoding;
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

/// Bound one chunk's write against the allocated state vector.
///
/// Returns the sample offset reached after writing `samples_in_chunk` samples
/// starting at `global_sample_offset`, or an error if that would run past the
/// `num_samples` the output buffer was allocated for.
///
/// This is the authoritative guard against the out-of-bounds device write
/// (CWE-787) — not the metadata reconciliation in `crate::readers::parquet`.
/// arrow-rs bounds decoding by a column chunk's byte range rather than by the
/// declared row counts, so a file whose footer and row-group counts lie
/// consistently passes that reconciliation and arrives here with more samples
/// than the allocation can hold.
pub(crate) fn ensure_within_capacity(
    global_sample_offset: usize,
    samples_in_chunk: usize,
    num_samples: usize,
) -> Result<usize> {
    let next_sample_offset = global_sample_offset
        .checked_add(samples_in_chunk)
        .ok_or_else(|| {
            MahoutError::MemoryAllocation(format!(
                "Sample offset overflow: {} + {}",
                global_sample_offset, samples_in_chunk
            ))
        })?;
    if next_sample_offset > num_samples {
        return Err(MahoutError::InvalidInput(format!(
            "Decoded sample count exceeds allocated capacity: writing {} samples at offset {} \
             would exceed the {} samples allocated (possible malformed Parquet metadata)",
            samples_in_chunk, global_sample_offset, num_samples
        )));
    }
    Ok(next_sample_offset)
}

/// Shared streaming pipeline for encoding data from Parquet files.
///
/// This function handles all the common IO, buffering, and GPU memory
/// management logic. The actual encoding is delegated to the `ChunkEncoder`.
///
/// # Null handling
///
/// The streaming Parquet path always uses [`crate::reader::NullHandling::FillZero`]
/// when constructing the [`crate::io::ParquetBlockReader`]. This replaces any
/// null values in the input with `0.0`, matching Mahout's historical behavior
/// and keeping the API backward compatible. Callers that require stricter
/// validation should ensure the input data contains no nulls.
pub(crate) fn stream_encode<E: ChunkEncoder>(
    engine: &QdpEngine,
    path: &str,
    num_qubits: usize,
    encoder: E,
) -> Result<*mut DLManagedTensor> {
    // Initialize reader
    let mut reader_core =
        crate::io::ParquetBlockReader::new(path, None, crate::reader::NullHandling::FillZero)?;
    let num_samples = reader_core.total_rows;

    // Allocate output state vector
    let total_state_vector = GpuStateVector::new_batch(
        &engine.device,
        num_samples,
        num_qubits,
        crate::Precision::Float64,
    )?;
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

    // Create cleanup guard to ensure resources are released on early return.
    // It takes ownership of our channel endpoints so that they are closed
    // before the IO thread is joined (see `shutdown_io_thread`).
    let cleanup_guard = CleanupGuard::new(&engine.device, io_handle, empty_buf_tx, full_buf_rx);

    // Main processing loop
    let mut global_sample_offset: usize = 0;
    let mut use_dev_a = true;

    loop {
        let (host_buffer, current_len) = match cleanup_guard.full_buf_rx().recv() {
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

        // Authoritative bounds check: the reader's row count is only a
        // consistency-checked hint, so every chunk is bounded here before any
        // kernel writes to the device buffer.
        let next_sample_offset =
            ensure_within_capacity(global_sample_offset, samples_in_chunk, num_samples)?;

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

                // Async copy to device (only if staging buffers are allocated).
                // `current_len` counts f64 elements; the copy takes bytes.
                if dev_staging.is_some() {
                    let copy_bytes = current_len
                        .checked_mul(std::mem::size_of::<f64>())
                        .ok_or_else(|| {
                            MahoutError::MemoryAllocation(format!(
                                "Staging copy size overflow: {} * {}",
                                current_len,
                                std::mem::size_of::<f64>()
                            ))
                        })?;
                    ctx.async_copy_to_device(
                        host_buffer.ptr() as *const c_void,
                        dev_ptr as *mut c_void,
                        copy_bytes,
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

            global_sample_offset = next_sample_offset;
            use_dev_a = !use_dev_a;
        }

        let _ = cleanup_guard.empty_buf_tx().send(host_buffer);
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

    // The device buffer was sized for `num_samples` from the file's declared row
    // count, and `GpuStateVector::new_batch` allocates without zeroing. If the
    // reader produced fewer samples, the uninitialized tail would be handed to
    // the caller through DLPack, so refuse instead of leaking GPU memory.
    //
    // A short read is not always the file's fault: `ParquetStreamingReader`
    // currently drops leading all-null rows in a `List` column under
    // `NullHandling::FillZero`, because the sample size is not yet known
    // at that point. See issue #1574; the error below hints at it.
    if global_sample_offset != num_samples {
        return Err(MahoutError::InvalidInput(format!(
            "Parquet file produced {} samples but its metadata declared {}; refusing to return \
             a partially initialized tensor (this can also happen when the file begins with \
             all-null rows in a List column)",
            global_sample_offset, num_samples
        )));
    }

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
    let encoding = Encoding::from_str_ci(encoding_method)?;
    match encoding {
        Encoding::Amplitude => {
            crate::profile_scope!("Mahout::EncodeAmplitudeFromParquet");
            stream_encode(engine, path, num_qubits, amplitude::AmplitudeEncoder)
        }
        Encoding::Angle => {
            crate::profile_scope!("Mahout::EncodeAngleFromParquet");
            stream_encode(engine, path, num_qubits, angle::AngleEncoder)
        }
        Encoding::Basis => {
            crate::profile_scope!("Mahout::EncodeBasisFromParquet");
            stream_encode(engine, path, num_qubits, basis::BasisEncoder)
        }
        _ => Err(MahoutError::NotImplemented(format!(
            "Encoding method '{}' not supported for streaming",
            encoding.as_str()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{RecvTimeoutError, sync_channel};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn ensure_within_capacity_allows_exact_fit() {
        // Chunks that exactly fill the allocation are the normal case.
        assert_eq!(ensure_within_capacity(0, 2, 4).unwrap(), 2);
        assert_eq!(ensure_within_capacity(2, 2, 4).unwrap(), 4);
        // A zero-sample chunk never moves the offset.
        assert_eq!(ensure_within_capacity(4, 0, 4).unwrap(), 4);
    }

    #[test]
    fn ensure_within_capacity_rejects_overrun() {
        // A Parquet file whose footer and row-group counts lie *consistently*
        // passes the reader's reconciliation and reaches here with more samples
        // than were allocated. This check is what stops the OOB device write.
        let err = ensure_within_capacity(0, 10_000, 1)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("exceeds allocated capacity"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn ensure_within_capacity_rejects_overrun_mid_stream() {
        // The overrun may only appear on a later chunk, once earlier chunks have
        // already advanced the offset.
        let err = ensure_within_capacity(3, 2, 4).unwrap_err().to_string();
        assert!(
            err.contains("exceeds allocated capacity"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn ensure_within_capacity_rejects_offset_overflow() {
        let err = ensure_within_capacity(usize::MAX, 1, usize::MAX)
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflow"), "unexpected error: {}", err);
    }

    /// Run `shutdown_io_thread` off the test thread so a regression shows up as
    /// a failure instead of hanging the test binary. Panics with a message that
    /// tells a deadlock apart from a panic inside the shutdown itself.
    fn expect_shutdown_completes(
        what: &str,
        f: impl FnOnce() -> thread::Result<()> + Send + 'static,
    ) {
        let (done_tx, done_rx) = sync_channel(1);
        thread::spawn(move || {
            let _ = done_tx.send(f().is_ok());
        });
        match done_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(true) => {}
            Ok(false) => panic!("{what}: the IO thread panicked"),
            Err(RecvTimeoutError::Timeout) => {
                panic!("{what}: shutdown deadlocked (did not complete within 10s)")
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("{what}: shutdown_io_thread itself panicked")
            }
        }
    }

    /// The IO thread parks in `empty_buf_rx.recv()`; only dropping our sender
    /// wakes it up, so the shutdown must drop it before joining.
    #[test]
    fn shutdown_unblocks_thread_waiting_for_empty_buffer() {
        let (full_buf_tx, full_buf_rx) = sync_channel::<usize>(2);
        let (empty_buf_tx, empty_buf_rx) = sync_channel::<usize>(2);

        let io_handle = thread::spawn(move || {
            while let Ok(buffer) = empty_buf_rx.recv() {
                if full_buf_tx.send(buffer).is_err() {
                    break;
                }
            }
        });

        let shutdown = move || shutdown_io_thread(Some(empty_buf_tx), Some(full_buf_rx), io_handle);
        expect_shutdown_completes("IO thread waiting for an empty buffer", shutdown);
    }

    /// The IO thread is blocked in `full_buf_tx.send()` with nobody draining
    /// the channel; only dropping our receiver wakes it up.
    #[test]
    fn shutdown_unblocks_thread_blocked_on_send() {
        // Capacity 0 makes the block deterministic: a rendezvous `send` cannot
        // complete until someone receives, and nobody here ever will. Avoids
        // depending on a sleep to land the thread inside `send()`.
        let (full_buf_tx, full_buf_rx) = sync_channel::<usize>(0);
        let (empty_buf_tx, empty_buf_rx) = sync_channel::<usize>(2);

        let io_handle = thread::spawn(move || {
            // Blocks immediately; only woken by `full_buf_rx` being dropped.
            let _ = full_buf_tx.send(0);
            drop(empty_buf_rx);
        });

        let shutdown = move || shutdown_io_thread(Some(empty_buf_tx), Some(full_buf_rx), io_handle);
        expect_shutdown_completes("IO thread blocked in send()", shutdown);
    }
}
