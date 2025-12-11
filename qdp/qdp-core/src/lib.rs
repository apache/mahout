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

pub mod dlpack;
pub mod gpu;
pub mod error;
pub mod preprocessing;
pub mod io;

#[macro_use]
mod profiling;

pub use error::{MahoutError, Result};

use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::ffi::c_void;
#[cfg(target_os = "linux")]
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
#[cfg(target_os = "linux")]
use std::thread;

use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
use crate::dlpack::DLManagedTensor;
use crate::gpu::get_encoder;
#[cfg(target_os = "linux")]
use crate::gpu::memory::{PinnedBuffer, GpuStateVector};
#[cfg(target_os = "linux")]
use crate::gpu::PipelineContext;
#[cfg(target_os = "linux")]
use qdp_kernels::{launch_l2_norm_batch, launch_amplitude_encode_batch};

/// 512MB staging buffer for large Parquet row groups (reduces fragmentation)
#[cfg(target_os = "linux")]
const STAGE_SIZE_BYTES: usize = 512 * 1024 * 1024;
#[cfg(target_os = "linux")]
const STAGE_SIZE_ELEMENTS: usize = STAGE_SIZE_BYTES / std::mem::size_of::<f64>();

/// Main entry point for Mahout QDP
///
/// Manages GPU context and dispatches encoding tasks.
/// Provides unified interface for device management, memory allocation, and DLPack.
pub struct QdpEngine {
    device: Arc<CudaDevice>,
}

impl QdpEngine {
    /// Initialize engine on GPU device
    ///
    /// # Arguments
    /// * `device_id` - CUDA device ID (typically 0)
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| MahoutError::Cuda(format!("Failed to initialize CUDA device {}: {:?}", device_id, e)))?;

        #[cfg(target_os = "linux")]
        {
            Ok(Self { device })
        }

        #[cfg(not(target_os = "linux"))]
        {
            Ok(Self { device })
        }
    }

    /// Encode classical data into quantum state
    ///
    /// Selects encoding strategy, executes on GPU, returns DLPack pointer.
    ///
    /// # Arguments
    /// * `data` - Input data
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// DLPack pointer for zero-copy PyTorch integration
    ///
    /// # Safety
    /// Pointer freed by DLPack deleter, do not free manually.
    pub fn encode(
        &self,
        data: &[f64],
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::Encode");

        let encoder = get_encoder(encoding_method)?;
        let state_vector = encoder.encode(&self.device, data, num_qubits)?;
        Ok(state_vector.to_dlpack())
    }

    /// Get CUDA device reference for advanced operations
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Encode multiple samples in a single fused kernel (most efficient)
    ///
    /// Allocates one large GPU buffer and launches a single batch kernel.
    /// This is faster than encode_batch() as it reduces allocation and kernel launch overhead.
    ///
    /// # Arguments
    /// * `batch_data` - Flattened batch data (all samples concatenated)
    /// * `num_samples` - Number of samples in the batch
    /// * `sample_size` - Size of each sample
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy (currently only "amplitude" supported for batch)
    ///
    /// # Returns
    /// Single DLPack pointer containing all encoded states (shape: [num_samples, 2^num_qubits])
    pub fn encode_batch(
        &self,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeBatch");

        let encoder = get_encoder(encoding_method)?;
        let state_vector = encoder.encode_batch(
            &self.device,
            batch_data,
            num_samples,
            sample_size,
            num_qubits,
        )?;

        let dlpack_ptr = state_vector.to_dlpack();
        Ok(dlpack_ptr)
    }

    /// Streaming Parquet encoder with multi-threaded IO
    ///
    /// Uses Producer-Consumer pattern: IO thread reads Parquet while GPU processes data.
    /// Double-buffered (ping-pong) for maximum pipeline overlap.
    ///
    /// # Arguments
    /// * `path` - Path to Parquet file with List<Float64> column
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Currently only "amplitude" supported for streaming
    ///
    /// # Returns
    /// DLPack pointer to encoded states [num_samples, 2^num_qubits]
    pub fn encode_from_parquet(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromParquet");

        #[cfg(target_os = "linux")]
        {
            if encoding_method != "amplitude" {
                return Err(MahoutError::NotImplemented("Only amplitude encoding supported for streaming".into()));
            }

            // Initialize reader
            let mut reader_core = crate::io::ParquetBlockReader::new(path)?;
            let num_samples = reader_core.total_samples;

            // Allocate GPU memory once
            let total_state_vector = GpuStateVector::new_batch(&self.device, num_samples, num_qubits)?;

            // Initialize dual-stream pipeline context
            let ctx = PipelineContext::new(&self.device)?;

            // Double-buffered device input (ping-pong)
            let dev_in_a = unsafe { self.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
            let dev_in_b = unsafe { self.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;

            // Setup Producer-Consumer channels
            let (full_buf_tx, full_buf_rx): (SyncSender<(PinnedBuffer, usize)>, Receiver<(PinnedBuffer, usize)>) = sync_channel(2);
            let (empty_buf_tx, empty_buf_rx): (SyncSender<PinnedBuffer>, Receiver<PinnedBuffer>) = sync_channel(2);

            // CRITICAL FIX: Pre-read first chunk to determine sample_size
            // This data must be processed, not discarded!
            let mut host_buf_first = PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?;
            let first_len = reader_core.read_chunk(host_buf_first.as_slice_mut())?;

            let sample_size = reader_core.get_sample_size()
                .ok_or_else(|| MahoutError::InvalidInput("Could not determine sample size".into()))?;

            // Send first chunk directly to GPU loop (must be processed first)
            full_buf_tx.send((host_buf_first, first_len))
                .map_err(|_| MahoutError::Io("Failed to send first buffer".into()))?;

            // Send one empty buffer to IO thread for subsequent reads
            empty_buf_tx.send(PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?)
                .map_err(|_| MahoutError::Io("Failed to send second buffer".into()))?;

            // Spawn IO thread (Producer): continues reading from second chunk onwards
            let mut reader = reader_core;
            let io_handle = thread::spawn(move || {
                loop {
                    let mut buffer = match empty_buf_rx.recv() {
                        Ok(b) => b,
                        Err(_) => break,
                    };

                    let len = match reader.read_chunk(buffer.as_slice_mut()) {
                        Ok(l) => l,
                        Err(e) => { eprintln!("IO Error: {:?}", e); 0 }
                    };

                    if full_buf_tx.send((buffer, len)).is_err() { break; }
                    if len == 0 { break; }
                }
            });

            // GPU processing loop: receives pre-read chunk, then IO thread chunks
            let mut global_sample_offset = 0;
            let mut use_dev_a = true;
            let state_len_per_sample = 1 << num_qubits;

            loop {
                let (host_buffer, current_len) = full_buf_rx.recv()
                    .map_err(|_| MahoutError::Io("IO thread disconnected".into()))?;

                // len == 0 means IO thread finished (don't recycle buffer)
                if current_len == 0 { break; }

                let samples_in_chunk = current_len / sample_size;
                if samples_in_chunk > 0 {
                    let dev_ptr = if use_dev_a { *dev_in_a.device_ptr() } else { *dev_in_b.device_ptr() };

                    unsafe {
                        crate::profile_scope!("GPU::Dispatch");

                        // Async H2D copy → record event → wait for copy → launch kernel
                        ctx.async_copy_to_device(&host_buffer, dev_ptr as *mut c_void, current_len);
                        ctx.record_copy_done();
                        ctx.wait_for_copy();

                        // Compute norms and encode batch
                        {
                            crate::profile_scope!("GPU::BatchEncode");
                            let offset_elements = global_sample_offset * state_len_per_sample;
                            let state_ptr_offset = total_state_vector.ptr().cast::<u8>()
                                .add(offset_elements * std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
                                .cast::<std::ffi::c_void>();

                            // Allocate norm buffer for this chunk
                            let mut norm_buffer = self.device.alloc_zeros::<f64>(samples_in_chunk)
                                .map_err(|e| MahoutError::MemoryAllocation(format!("Failed to allocate norm buffer: {:?}", e)))?;

                            // Step 1: Compute L2 norms for this chunk
                            {
                                crate::profile_scope!("GPU::NormBatch");
                                let ret = launch_l2_norm_batch(
                                    dev_ptr as *const f64,
                                    samples_in_chunk,
                                    sample_size,
                                    *norm_buffer.device_ptr_mut() as *mut f64,
                                    ctx.stream_compute.stream as *mut c_void
                                );
                                if ret != 0 {
                                    return Err(MahoutError::KernelLaunch(format!("Norm kernel error: {}", ret)));
                                }
                            }

                            // Step 2: Encode batch using computed norms
                            {
                                crate::profile_scope!("GPU::EncodeBatch");
                                let ret = launch_amplitude_encode_batch(
                                    dev_ptr as *const f64,
                                    state_ptr_offset,
                                    *norm_buffer.device_ptr() as *const f64,
                                    samples_in_chunk,
                                    sample_size,
                                    state_len_per_sample,
                                    ctx.stream_compute.stream as *mut c_void
                                );
                                if ret != 0 {
                                    return Err(MahoutError::KernelLaunch(format!("Encode kernel error: {}", ret)));
                                }
                            }
                        }

                        // Sync copy stream before buffer reuse
                        ctx.sync_copy_stream();
                    }
                    global_sample_offset += samples_in_chunk;
                    use_dev_a = !use_dev_a;
                }

                // Return buffer to IO thread (ignore errors if thread exited)
                let _ = empty_buf_tx.send(host_buffer);
            }

            self.device.synchronize().map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
            let _ = io_handle.join();

            // Transfer ownership to DLPack (Arc handles ref counting)
            let dlpack_ptr = total_state_vector.to_dlpack();
            Ok(dlpack_ptr)
        }

        #[cfg(not(target_os = "linux"))]
        {
            let (batch_data, num_samples, sample_size) = crate::io::read_parquet_batch(path)?;
            self.encode_batch(&batch_data, num_samples, sample_size, num_qubits, encoding_method)
        }
    }

    /// Load data from Arrow IPC file and encode into quantum state
    ///
    /// Supports:
    /// - FixedSizeList<Float64> - fastest, all samples same size
    /// - List<Float64> - flexible, variable sample sizes
    ///
    /// # Arguments
    /// * `path` - Path to Arrow IPC file (.arrow or .feather)
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// Single DLPack pointer containing all encoded states (shape: [num_samples, 2^num_qubits])
    pub fn encode_from_arrow_ipc(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromArrowIPC");

        // Read Arrow IPC (6x faster than Parquet)
        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadArrowIPCBatch");
            crate::io::read_arrow_ipc_batch(path)?
        };

        // Encode using fused batch kernel
        self.encode_batch(&batch_data, num_samples, sample_size, num_qubits, encoding_method)
    }
}

// Re-export key types for convenience
pub use gpu::QuantumEncoder;
