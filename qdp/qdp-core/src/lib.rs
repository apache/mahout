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

use std::sync::{Arc, Mutex};
#[cfg(target_os = "linux")]
use std::ffi::c_void;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, safe::CudaStream};
use crate::dlpack::DLManagedTensor;
use crate::gpu::get_encoder;
#[cfg(target_os = "linux")]
use crate::gpu::memory::{PinnedBuffer, GpuStateVector};
#[cfg(target_os = "linux")]
use qdp_kernels::launch_amplitude_encode_batch;

#[cfg(target_os = "linux")]
const STAGE_SIZE_BYTES: usize = 64 * 1024 * 1024;
#[cfg(target_os = "linux")]
const STAGE_SIZE_ELEMENTS: usize = STAGE_SIZE_BYTES / std::mem::size_of::<f64>();

#[cfg(target_os = "linux")]
struct PipelineResources {
    pinned_a: PinnedBuffer,
    pinned_b: PinnedBuffer,
    dev_in_a: CudaSlice<f64>,
    dev_in_b: CudaSlice<f64>,
    norms_pinned_a: PinnedBuffer,
    norms_pinned_b: PinnedBuffer,
    dev_norms_a: CudaSlice<f64>,
    dev_norms_b: CudaSlice<f64>,
    stream_a: CudaStream,
    stream_b: CudaStream,
}

#[cfg(target_os = "linux")]
unsafe impl Send for PipelineResources {}
#[cfg(target_os = "linux")]
unsafe impl Sync for PipelineResources {}

/// Main entry point for Mahout QDP
///
/// Manages GPU context and dispatches encoding tasks.
/// Provides unified interface for device management, memory allocation, and DLPack.
pub struct QdpEngine {
    device: Arc<CudaDevice>,
    #[cfg(target_os = "linux")]
    resources: Mutex<PipelineResources>,
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
            let pinned_a = PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?;
            let pinned_b = PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?;
            let norms_pinned_a = PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?;
            let norms_pinned_b = PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?;

            let dev_in_a = unsafe { device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
            let dev_in_b = unsafe { device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
            let dev_norms_a = unsafe { device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
            let dev_norms_b = unsafe { device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;

            let stream_a = device.fork_default_stream()
                .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
            let stream_b = device.fork_default_stream()
                .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;

            Ok(Self {
                device,
                resources: Mutex::new(PipelineResources {
                    pinned_a,
                    pinned_b,
                    dev_in_a,
                    dev_in_b,
                    norms_pinned_a,
                    norms_pinned_b,
                    dev_norms_a,
                    dev_norms_b,
                    stream_a,
                    stream_b,
                }),
            })
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

            const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;

            let min_required_elements = 1 << num_qubits;
            if STAGE_SIZE_ELEMENTS < min_required_elements {
                return Err(MahoutError::MemoryAllocation(format!(
                    "Staging buffer ({} MB) is too small for {} qubits.",
                    STAGE_SIZE_BYTES / (1024*1024),
                    num_qubits
                )));
            }

            let mut res = self.resources.lock()
                .map_err(|_| MahoutError::Cuda("Failed to lock pipeline resources".into()))?;

            let mut reader = crate::io::ParquetBlockReader::new(path)?;

            let read_len = reader.read_chunk(res.pinned_a.as_slice_mut())?;
            let sample_size = reader.get_sample_size()
                .ok_or_else(|| MahoutError::InvalidInput("Could not determine sample size".into()))?;
            let num_samples = reader.total_samples;

            let total_state_vector = GpuStateVector::new_batch(&self.device, num_samples, num_qubits)?;

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

                        if global_sample_offset + samples_in_chunk > num_samples {
                            return Err(MahoutError::InvalidInput(
                                "Parquet file contains more data than metadata".into()
                            ));
                        }

                        {
                            crate::profile_scope!("CPU::CalcNorms");
                            let (data_slice, norms_ptr) = if use_buffer_a {
                                let norms = res.norms_pinned_a.ptr();
                                (&res.pinned_a.as_slice_mut()[0..current_read_len], norms)
                            } else {
                                let norms = res.norms_pinned_b.ptr();
                                (&res.pinned_b.as_slice_mut()[0..current_read_len], norms)
                            };
                            let target_norms = std::slice::from_raw_parts_mut(
                                norms_ptr as *mut f64,
                                samples_in_chunk
                            );

                            use rayon::prelude::*;
                            target_norms.par_iter_mut().enumerate().for_each(|(i, norm_out)| {
                                let start = i * sample_size;
                                let sample = &data_slice[start..start + sample_size];
                                *norm_out = 1.0 / sample.iter().map(|x| x*x).sum::<f64>().sqrt();
                            });
                        }

                        {
                            crate::profile_scope!("GPU::CopyData");
                            let (pinned_buf, norms_buf, dev_in, dev_norms, stream) = if use_buffer_a {
                                (&res.pinned_a, &res.norms_pinned_a, &res.dev_in_a, &res.dev_norms_a, &res.stream_a)
                            } else {
                                (&res.pinned_b, &res.norms_pinned_b, &res.dev_in_b, &res.dev_norms_b, &res.stream_b)
                            };
                            cudaMemcpyAsync(
                                *dev_in.device_ptr() as *mut c_void,
                                pinned_buf.ptr() as *const c_void,
                                current_read_len * std::mem::size_of::<f64>(),
                                CUDA_MEMCPY_HOST_TO_DEVICE,
                                stream.stream as *mut c_void
                            );
                            cudaMemcpyAsync(
                                *dev_norms.device_ptr() as *mut c_void,
                                norms_buf.ptr() as *const c_void,
                                samples_in_chunk * std::mem::size_of::<f64>(),
                                CUDA_MEMCPY_HOST_TO_DEVICE,
                                stream.stream as *mut c_void
                            );
                        }

                        {
                            crate::profile_scope!("GPU::Kernel");
                            let offset_elements = global_sample_offset * state_len_per_sample;
                            let state_ptr_offset = total_state_vector.ptr().cast::<u8>()
                                .add(offset_elements * std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
                                .cast::<std::ffi::c_void>();

                            let (dev_in, dev_norms, stream) = if use_buffer_a {
                                (&res.dev_in_a, &res.dev_norms_a, &res.stream_a)
                            } else {
                                (&res.dev_in_b, &res.dev_norms_b, &res.stream_b)
                            };

                            let ret = launch_amplitude_encode_batch(
                                *dev_in.device_ptr() as *const f64,
                                state_ptr_offset,
                                *dev_norms.device_ptr() as *const f64,
                                samples_in_chunk,
                                sample_size,
                                state_len_per_sample,
                                stream.stream as *mut c_void
                            );

                            if ret != 0 {
                                return Err(MahoutError::KernelLaunch(format!("Error: {}", ret)));
                            }
                        }

                        global_sample_offset += samples_in_chunk;
                    }

                    {
                        crate::profile_scope!("IO::ReadNext");
                        let next_stream = if use_buffer_a { &res.stream_b } else { &res.stream_a };
                        self.device.wait_for(next_stream)
                            .map_err(|e| MahoutError::Cuda(format!("Sync error: {:?}", e)))?;
                        use_buffer_a = !use_buffer_a;
                        let next_pinned = if use_buffer_a { &mut res.pinned_a } else { &mut res.pinned_b };
                        current_read_len = reader.read_chunk(next_pinned.as_slice_mut())?;
                    }
                }
            }

            self.device.synchronize().map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;

            Ok(total_state_vector.to_dlpack())
        }

        #[cfg(not(target_os = "linux"))]
        {
            let (batch_data, num_samples, sample_size) = crate::io::read_parquet_batch(path)?;
            self.encode_batch(&batch_data, num_samples, sample_size, num_qubits, encoding_method)
        }
    }
}

// Re-export key types for convenience
pub use gpu::QuantumEncoder;
