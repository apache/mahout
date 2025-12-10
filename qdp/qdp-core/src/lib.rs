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

use cudarc::driver::{CudaDevice, DevicePtr};
use crate::dlpack::DLManagedTensor;
use crate::gpu::get_encoder;
#[cfg(target_os = "linux")]
use crate::gpu::memory::{PinnedBuffer, GpuStateVector};
#[cfg(target_os = "linux")]
use qdp_kernels::launch_amplitude_encode_batch;

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
        Ok(Self {
            device  // CudaDevice::new already returns Arc<CudaDevice> in cudarc 0.11
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

        let dlpack_ptr = state_vector.to_dlpack();
        Ok(dlpack_ptr)
    }

    /// Load data from Parquet file and encode into quantum state
    ///
    /// Reads Parquet file with List<Float64> column format and encodes all samples
    /// in a single batch operation. Bypasses pandas for maximum performance.
    ///
    /// # Arguments
    /// * `path` - Path to Parquet file
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// Single DLPack pointer containing all encoded states (shape: [num_samples, 2^num_qubits])
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
                return Err(MahoutError::NotImplemented("Only amplitude encoding supported for parquet streaming".into()));
            }

            const STAGE_SIZE_BYTES: usize = 64 * 1024 * 1024; // 64MB per stage
            const STAGE_SIZE_ELEMENTS: usize = STAGE_SIZE_BYTES / std::mem::size_of::<f64>();
            const BYTES_PER_F64: usize = 8;
            const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;

            let mut reader = crate::io::ParquetBlockReader::new(path)?;

            // Setup double-buffered staging area
            let mut pinned_a = PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?;
            let mut pinned_b = PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?;
            let dev_in_a = unsafe { self.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
            let dev_in_b = unsafe { self.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;

            // Pre-read first chunk to determine sample size
            let read_len = reader.read_chunk(pinned_a.as_slice_mut())?;
            let sample_size = reader.get_sample_size()
                .ok_or_else(|| MahoutError::InvalidInput("Could not determine sample size from parquet".into()))?;
            let num_samples = reader.total_samples;

            let max_samples_per_chunk = STAGE_SIZE_ELEMENTS / sample_size + 1;
            let norms_pinned_a = PinnedBuffer::new(max_samples_per_chunk)?;
            let norms_pinned_b = PinnedBuffer::new(max_samples_per_chunk)?;
            let dev_norms_a = unsafe { self.device.alloc::<f64>(max_samples_per_chunk) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
            let dev_norms_b = unsafe { self.device.alloc::<f64>(max_samples_per_chunk) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;

            let total_state_vector = GpuStateVector::new_batch(&self.device, num_samples, num_qubits)?;
            let stream_a = self.device.fork_default_stream()
                .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
            let stream_b = self.device.fork_default_stream()
                .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;

            let mut global_sample_offset = 0;
            let mut current_read_len = read_len;
            let mut use_buffer_a = true;
            let state_len_per_sample = 1 << num_qubits;

            unsafe {
                unsafe extern "C" {
                    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: u32, stream: *mut c_void) -> i32;
                }

                loop {
                    if current_read_len == 0 { break; }

                    let samples_in_chunk = current_read_len / sample_size;

                    if samples_in_chunk > 0 {
                        crate::profile_scope!("Pipeline::Step");

                        // CPU: Calculate norms (overlapped with GPU work)
                        {
                            crate::profile_scope!("CPU::CalcNorms");
                            let (data_slice, norms_slice) = if use_buffer_a {
                                (
                                    &pinned_a.as_slice_mut()[0..current_read_len],
                                    std::slice::from_raw_parts_mut(norms_pinned_a.ptr() as *mut f64, samples_in_chunk)
                                )
                            } else {
                                (
                                    &pinned_b.as_slice_mut()[0..current_read_len],
                                    std::slice::from_raw_parts_mut(norms_pinned_b.ptr() as *mut f64, samples_in_chunk)
                                )
                            };

                            use rayon::prelude::*;
                            norms_slice.par_iter_mut().enumerate().for_each(|(i, norm_out)| {
                                let start = i * sample_size;
                                let sample = &data_slice[start..start + sample_size];
                                *norm_out = 1.0 / sample.iter().map(|x| x*x).sum::<f64>().sqrt();
                            });
                        }

                        // GPU: Async copy data and norms
                        {
                            crate::profile_scope!("GPU::CopyData");
                            let (pinned_ptr, norms_ptr, dev_in_ptr, dev_norms_ptr, stream_ptr) = if use_buffer_a {
                                (
                                    pinned_a.ptr() as *const c_void,
                                    norms_pinned_a.ptr() as *const c_void,
                                    *dev_in_a.device_ptr() as *mut c_void,
                                    *dev_norms_a.device_ptr() as *mut c_void,
                                    stream_a.stream as *mut c_void
                                )
                            } else {
                                (
                                    pinned_b.ptr() as *const c_void,
                                    norms_pinned_b.ptr() as *const c_void,
                                    *dev_in_b.device_ptr() as *mut c_void,
                                    *dev_norms_b.device_ptr() as *mut c_void,
                                    stream_b.stream as *mut c_void
                                )
                            };

                            cudaMemcpyAsync(dev_in_ptr, pinned_ptr, current_read_len * BYTES_PER_F64, CUDA_MEMCPY_HOST_TO_DEVICE, stream_ptr);
                            cudaMemcpyAsync(dev_norms_ptr, norms_ptr, samples_in_chunk * BYTES_PER_F64, CUDA_MEMCPY_HOST_TO_DEVICE, stream_ptr);
                        }

                        // GPU: Launch kernel
                        {
                            crate::profile_scope!("GPU::Kernel");
                            let offset_elements = global_sample_offset * state_len_per_sample;
                            let state_ptr_offset = total_state_vector.ptr().cast::<u8>()
                                .add(offset_elements * std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
                                .cast::<std::ffi::c_void>();

                            let (dev_in_ptr, dev_norms_ptr, stream_ptr) = if use_buffer_a {
                                (*dev_in_a.device_ptr() as *const f64, *dev_norms_a.device_ptr() as *const f64, stream_a.stream as *mut c_void)
                            } else {
                                (*dev_in_b.device_ptr() as *const f64, *dev_norms_b.device_ptr() as *const f64, stream_b.stream as *mut c_void)
                            };

                            let ret = launch_amplitude_encode_batch(
                                dev_in_ptr, state_ptr_offset, dev_norms_ptr,
                                samples_in_chunk, sample_size, state_len_per_sample, stream_ptr
                            );

                            if ret != 0 {
                                return Err(MahoutError::KernelLaunch(format!("Pipeline kernel failed: {}", ret)));
                            }
                        }

                        global_sample_offset += samples_in_chunk;
                    }

                    // CPU: Read next chunk (overlapped with GPU work)
                    {
                        crate::profile_scope!("IO::ReadNext");

                        // Sync: Wait for next stream to finish before overwriting its buffer
                        // Prevents data race where CPU writes while GPU DMA reads
                        let next_stream = if use_buffer_a { &stream_b } else { &stream_a };
                        self.device.wait_for(next_stream)
                            .map_err(|e| MahoutError::Cuda(format!("Stream sync failed before IO: {:?}", e)))?;

                        use_buffer_a = !use_buffer_a;
                        let next_pinned = if use_buffer_a { &mut pinned_a } else { &mut pinned_b };
                        current_read_len = reader.read_chunk(next_pinned.as_slice_mut())?;
                    }
                }
            }

            // Sync everything at the end
            self.device.synchronize().map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;

            Ok(total_state_vector.to_dlpack())
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for non-Linux
            // Read Parquet directly using Arrow (faster than pandas)
            let (batch_data, num_samples, sample_size) = {
                crate::profile_scope!("IO::ReadParquetBatch");
                crate::io::read_parquet_batch(path)?
            };

            // Encode using fused batch kernel
            self.encode_batch(&batch_data, num_samples, sample_size, num_qubits, encoding_method)
        }
    }
}

// Re-export key types for convenience
pub use gpu::QuantumEncoder;
