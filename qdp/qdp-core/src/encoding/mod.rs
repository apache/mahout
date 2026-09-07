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

//! Streaming encode of a Parquet file.
//!
//! An IO thread reads 512 MB chunks into pinned host buffers while the main
//! thread copies the previous chunk to a device staging buffer and launches
//! the encoding kernel on it. Any [`Kernel`] works here: a chunk is just a
//! batch of whole samples written at an offset into one big state vector.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use cudarc::driver::{CudaDevice, DevicePtr as _};

use crate::dlpack::DLManagedTensor;
use crate::gpu::PipelineContext;
use crate::gpu::kernels::{DeviceInput, HostInput, Kernel, LaunchCtx, Output, Pending, Shape};
use crate::gpu::memory::{GpuStateVector, PinnedHostBuffer};
use crate::reader::StreamingDataReader;
use crate::types::Encoding;
use crate::{MahoutError, QdpEngine, Result};
use qdp_kernels::CuDoubleComplex;

pub(crate) const STAGE_SIZE_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const STAGE_SIZE_ELEMENTS: usize = STAGE_SIZE_BYTES / std::mem::size_of::<f64>();
/// Bound on the float64 staging state used when the engine precision is float32.
const STAGING_STATE_BYTES: usize = 256 * 1024 * 1024;

type FullBufferResult = std::result::Result<(PinnedHostBuffer, usize), MahoutError>;
type FullBufferChannel = (SyncSender<FullBufferResult>, Receiver<FullBufferResult>);

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

pub(crate) fn stream_encode(
    engine: &QdpEngine,
    path: &str,
    num_qubits: usize,
    kernel: &dyn Kernel,
) -> Result<*mut DLManagedTensor> {
    let device = engine.device();
    let mut reader_core =
        crate::io::ParquetBlockReader::new(path, None, crate::reader::NullHandling::FillZero)?;
    let num_samples = reader_core.total_rows;
    crate::gpu::kernels::validate_qubit_count(num_qubits)?;
    // The kernels write float64; when the engine wants float32 the result is
    // converted chunk by chunk through a bounded float64 staging state, so the
    // file never needs two full-size states resident at once.
    let precision = engine.precision();
    let state_len = 1usize << num_qubits;
    let total_state_vector = GpuStateVector::new_batch(device, num_samples, num_qubits, precision)?;
    let out = Output::of(&total_state_vector);
    let staging_samples = (STAGING_STATE_BYTES
        / (state_len * std::mem::size_of::<CuDoubleComplex>()))
    .clamp(1, num_samples.max(1));
    let staging = if precision == crate::Precision::Float32 && num_samples > 0 {
        Some(GpuStateVector::new_batch(
            device,
            staging_samples,
            num_qubits,
            crate::Precision::Float64,
        )?)
    } else {
        None
    };
    let mut pending = Pending::new();

    const PIPELINE_EVENT_SLOTS: usize = 2;
    let ctx = PipelineContext::new(device, PIPELINE_EVENT_SLOTS)?;
    let dev_in_a = unsafe { device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
        .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
    let dev_in_b = unsafe { device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
        .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;

    let (full_buf_tx, full_buf_rx): FullBufferChannel = sync_channel(2);
    let (empty_buf_tx, empty_buf_rx): (SyncSender<PinnedHostBuffer>, _) = sync_channel(2);

    let mut host_buf_first = PinnedHostBuffer::new(STAGE_SIZE_ELEMENTS)?;
    let first_len = reader_core.read_chunk(host_buf_first.as_slice_mut())?;
    let sample_size = reader_core
        .get_sample_size()
        .ok_or_else(|| MahoutError::InvalidInput("Could not determine sample size".into()))?;
    if sample_size == 0 {
        return Err(MahoutError::InvalidInput(
            "Streaming encode requires sample_size > 0".into(),
        ));
    }
    if sample_size > STAGE_SIZE_ELEMENTS {
        return Err(MahoutError::InvalidInput(format!(
            "Sample size {} exceeds staging buffer capacity {}",
            sample_size, STAGE_SIZE_ELEMENTS
        )));
    }
    kernel.validate_shape(Shape::new(num_samples.max(1), sample_size), num_qubits)?;

    full_buf_tx
        .send(Ok((host_buf_first, first_len)))
        .map_err(|_| MahoutError::Io("Failed to send first buffer".into()))?;
    empty_buf_tx
        .send(PinnedHostBuffer::new(STAGE_SIZE_ELEMENTS)?)
        .map_err(|_| MahoutError::Io("Failed to send second buffer".into()))?;

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
    let cleanup_guard = CleanupGuard::new(device, io_handle);

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
            let chunk = Shape::new(samples_in_chunk, sample_size);
            kernel
                .validate_host(
                    &HostInput::F64(&host_buffer.as_slice()[..current_len]),
                    chunk,
                    num_qubits,
                )
                .map_err(|e| {
                    MahoutError::InvalidInput(format!(
                        "{} (in chunk starting at sample {})",
                        crate::gpu::kernels::invalid_input_text(&e),
                        global_sample_offset
                    ))
                })?;

            let event_slot = if use_dev_a { 0 } else { 1 };
            let dev_ptr = if use_dev_a {
                *dev_in_a.device_ptr()
            } else {
                *dev_in_b.device_ptr()
            };
            unsafe {
                crate::profile_scope!("GPU::Dispatch");
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

                let launch_ctx = LaunchCtx::new(device, ctx.stream_compute.stream as *mut c_void);
                match &staging {
                    None => {
                        pending.merge(kernel.launch(
                            &launch_ctx,
                            DeviceInput::F64(dev_ptr as *const f64),
                            chunk,
                            num_qubits,
                            out.offset_samples(global_sample_offset)?,
                        )?);
                    }
                    Some(stage) => {
                        // Encode into the float64 staging state a slice at a
                        // time, converting each slice into the float32 result.
                        // Everything is queued on the compute stream, so the
                        // staging buffer is reused in order.
                        let stage_out = Output::of(stage);
                        let mut done = 0usize;
                        while done < samples_in_chunk {
                            let n = (samples_in_chunk - done).min(staging_samples);
                            let input = (dev_ptr as *const f64).add(done * sample_size);
                            pending.merge(kernel.launch(
                                &launch_ctx,
                                DeviceInput::F64(input),
                                Shape::new(n, sample_size),
                                num_qubits,
                                stage_out,
                            )?);
                            let len = n * state_len;
                            let dst = out.offset_samples(global_sample_offset + done)?.as_f32()?;
                            launch_ctx.launch(
                                "amplitude",
                                "convert_state_to_complex64_kernel",
                                qdp_kernels::LaunchConfig::grid_1d(len),
                                &mut qdp_kernels::kernel_args![stage_out.as_f64()?, dst, len],
                            )?;
                            done += n;
                        }
                    }
                }
                ctx.sync_copy_stream()?;
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

    let io_handle = cleanup_guard.defuse();
    device
        .synchronize()
        .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
    io_handle
        .join()
        .map_err(|e| MahoutError::Io(format!("IO thread panicked: {:?}", e)))?;
    // Streaming keeps the prior file semantics: a sample the kernel could not
    // normalise (all-null rows filled with zero) becomes a zero row rather
    // than failing the whole file, so deferred checks are dropped.
    pending.without_checks().finish(device)?;

    Ok(total_state_vector.to_dlpack())
}

pub(crate) fn encode_from_parquet(
    engine: &QdpEngine,
    path: &str,
    num_qubits: usize,
    encoding_method: &str,
) -> Result<*mut DLManagedTensor> {
    crate::profile_scope!("Mahout::EncodeFromParquet");
    let encoding = Encoding::from_str_ci(encoding_method)?;
    stream_encode(engine, path, num_qubits, encoding.encoder())
}
