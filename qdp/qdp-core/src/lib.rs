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
pub mod reader;
pub mod readers;
#[macro_use]
mod profiling;

pub use error::{MahoutError, Result};
pub use gpu::memory::Precision;

use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::ffi::c_void;
#[cfg(target_os = "linux")]
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
#[cfg(target_os = "linux")]
use std::thread;

use cudarc::driver::CudaDevice;
use crate::dlpack::DLManagedTensor;
use crate::gpu::get_encoder;
#[cfg(target_os = "linux")]
use crate::gpu::memory::{GpuStateVector, PinnedHostBuffer};
#[cfg(target_os = "linux")]
use crate::reader::StreamingDataReader;
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
    precision: Precision,
}

impl QdpEngine {
    /// Initialize engine on GPU device
    ///
    /// # Arguments
    /// * `device_id` - CUDA device ID (typically 0)
    pub fn new(device_id: usize) -> Result<Self> {
        Self::new_with_precision(device_id, Precision::Float32)
    }

    /// Initialize engine with explicit precision.
    pub fn new_with_precision(device_id: usize, precision: Precision) -> Result<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| MahoutError::Cuda(format!("Failed to initialize CUDA device {}: {:?}", device_id, e)))?;
        Ok(Self {
            device,  // CudaDevice::new already returns Arc<CudaDevice> in cudarc 0.11
            precision,
        })
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
        let state_vector = state_vector.to_precision(&self.device, self.precision)?;
        let dlpack_ptr = {
            crate::profile_scope!("DLPack::Wrap");
            state_vector.to_dlpack()
        };
        Ok(dlpack_ptr)
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

        let state_vector = state_vector.to_precision(&self.device, self.precision)?;
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

            let mut reader_core = crate::io::ParquetBlockReader::new(path, None)?;
            let num_samples = reader_core.total_rows;

            let total_state_vector = GpuStateVector::new_batch(&self.device, num_samples, num_qubits)?;
            const PIPELINE_EVENT_SLOTS: usize = 2; // matches double-buffered staging buffers
            let ctx = PipelineContext::new(&self.device, PIPELINE_EVENT_SLOTS)?;

            let dev_in_a = unsafe { self.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
            let dev_in_b = unsafe { self.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;

            let (full_buf_tx, full_buf_rx): (SyncSender<std::result::Result<(PinnedHostBuffer, usize), MahoutError>>, Receiver<std::result::Result<(PinnedHostBuffer, usize), MahoutError>>) = sync_channel(2);
            let (empty_buf_tx, empty_buf_rx): (SyncSender<PinnedHostBuffer>, Receiver<PinnedHostBuffer>) = sync_channel(2);

            let mut host_buf_first = PinnedHostBuffer::new(STAGE_SIZE_ELEMENTS)?;
            let first_len = reader_core.read_chunk(host_buf_first.as_slice_mut())?;

            let sample_size = reader_core.get_sample_size()
                .ok_or_else(|| MahoutError::InvalidInput("Could not determine sample size".into()))?;

            if sample_size == 0 {
                return Err(MahoutError::InvalidInput("Sample size cannot be zero".into()));
            }

            full_buf_tx.send(Ok((host_buf_first, first_len)))
                .map_err(|_| MahoutError::Io("Failed to send first buffer".into()))?;

            empty_buf_tx.send(PinnedHostBuffer::new(STAGE_SIZE_ELEMENTS)?)
                .map_err(|_| MahoutError::Io("Failed to send second buffer".into()))?;

            let mut reader = reader_core;
            let io_handle = thread::spawn(move || {
                loop {
                    let mut buffer = match empty_buf_rx.recv() {
                        Ok(b) => b,
                        Err(_) => break,
                    };

                    let result = reader.read_chunk(buffer.as_slice_mut()).map(|len| (buffer, len));

                    let should_break = match &result {
                        Ok((_, len)) => *len == 0,
                        Err(_) => true,
                    };

                    if full_buf_tx.send(result).is_err() { break; }

                    if should_break { break; }
                }
            });

            let mut global_sample_offset: usize = 0;
            let mut use_dev_a = true;
            let state_len_per_sample = 1 << num_qubits;

            loop {
                let (host_buffer, current_len) = match full_buf_rx.recv() {
                    Ok(Ok((buffer, len))) => (buffer, len),
                    Ok(Err(e)) => return Err(e),
                    Err(_) => return Err(MahoutError::Io("IO thread disconnected".into())),
                };

                if current_len == 0 { break; }

                if current_len % sample_size != 0 {
                    return Err(MahoutError::InvalidInput(format!(
                        "Chunk length {} is not a multiple of sample size {}",
                        current_len, sample_size
                    )));
                }

                let samples_in_chunk = current_len / sample_size;
                if samples_in_chunk > 0 {
                    let event_slot = if use_dev_a { 0 } else { 1 };
                    let dev_ptr = if use_dev_a { *dev_in_a.device_ptr() } else { *dev_in_b.device_ptr() };

                    unsafe {
                        crate::profile_scope!("GPU::Dispatch");

                        ctx.async_copy_to_device(host_buffer.ptr() as *const c_void, dev_ptr as *mut c_void, current_len)?;
                        ctx.record_copy_done(event_slot)?;
                        ctx.wait_for_copy(event_slot)?;

                        {
                            crate::profile_scope!("GPU::BatchEncode");
                            let offset_elements = global_sample_offset
                                .checked_mul(state_len_per_sample)
                                .ok_or_else(|| MahoutError::MemoryAllocation(
                                    format!("Offset calculation overflow: {} * {}", global_sample_offset, state_len_per_sample)
                                ))?;

                            let offset_bytes = offset_elements
                                .checked_mul(std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
                                .ok_or_else(|| MahoutError::MemoryAllocation(
                                    format!("Offset bytes calculation overflow: {} * {}", offset_elements, std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
                                ))?;

                            let state_ptr_offset = total_state_vector.ptr_void().cast::<u8>()
                                .add(offset_bytes)
                                .cast::<std::ffi::c_void>();

                            let mut norm_buffer = self.device.alloc_zeros::<f64>(samples_in_chunk)
                                .map_err(|e| MahoutError::MemoryAllocation(format!("Failed to allocate norm buffer: {:?}", e)))?;

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

                        ctx.sync_copy_stream()?;
                    }
                    global_sample_offset = global_sample_offset
                        .checked_add(samples_in_chunk)
                        .ok_or_else(|| MahoutError::MemoryAllocation(
                            format!("Sample offset overflow: {} + {}", global_sample_offset, samples_in_chunk)
                        ))?;
                    use_dev_a = !use_dev_a;
                }

                let _ = empty_buf_tx.send(host_buffer);
            }

            self.device.synchronize().map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
            io_handle.join().map_err(|e| MahoutError::Io(format!("IO thread panicked: {:?}", e)))?;

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

        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadArrowIPCBatch");
            crate::io::read_arrow_ipc_batch(path)?
        };

        self.encode_batch(&batch_data, num_samples, sample_size, num_qubits, encoding_method)
    }
}

// Re-export key types for convenience
pub use gpu::QuantumEncoder;
