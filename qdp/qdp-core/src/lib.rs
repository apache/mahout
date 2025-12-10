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
use qdp_kernels::launch_fused_amplitude_encode_batch;

#[cfg(target_os = "linux")]
const STAGE_SIZE_BYTES: usize = 256 * 1024 * 1024;  // 256MB buffer (reduces IO calls)
#[cfg(target_os = "linux")]
const STAGE_SIZE_ELEMENTS: usize = STAGE_SIZE_BYTES / std::mem::size_of::<f64>();

// CUDA FFI for event management
#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: u32, stream: *mut c_void) -> i32;
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
}

#[cfg(target_os = "linux")]
const CUDA_EVENT_DISABLE_TIMING: u32 = 0x02;

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

            // Allocate resources once (not in hot path)
            let mut host_buf_a = PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?;
            let mut host_buf_b = PinnedBuffer::new(STAGE_SIZE_ELEMENTS)?;

            let dev_in_a = unsafe { self.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;
            let dev_in_b = unsafe { self.device.alloc::<f64>(STAGE_SIZE_ELEMENTS) }
                .map_err(|e| MahoutError::MemoryAllocation(format!("{:?}", e)))?;

            let stream_compute = self.device.fork_default_stream()
                .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
            let stream_copy = self.device.fork_default_stream()
                .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;

            let mut reader = crate::io::ParquetBlockReader::new(path)?;
            let sample_size = reader.get_sample_size()
                .ok_or_else(|| MahoutError::InvalidInput("Could not determine sample size".into()))?;
            let num_samples = reader.total_samples;

            let total_state_vector = GpuStateVector::new_batch(&self.device, num_samples, num_qubits)?;

            let mut global_sample_offset = 0;
            let mut current_read_len = reader.read_chunk(host_buf_a.as_slice_mut())?;
            let mut use_buffer_a = true;
            let state_len_per_sample = 1 << num_qubits;

            // Create events for pipeline synchronization
            let mut event_copy_done_a: *mut c_void = std::ptr::null_mut();
            let mut event_copy_done_b: *mut c_void = std::ptr::null_mut();

            unsafe {
                let ret_a = cudaEventCreateWithFlags(&mut event_copy_done_a, CUDA_EVENT_DISABLE_TIMING);
                let ret_b = cudaEventCreateWithFlags(&mut event_copy_done_b, CUDA_EVENT_DISABLE_TIMING);
                if ret_a != 0 || ret_b != 0 {
                    return Err(MahoutError::Cuda(format!("Failed to create CUDA events: {} {}", ret_a, ret_b)));
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

                        // Select current resources and corresponding event
                        let (curr_host, curr_dev, curr_event) = if use_buffer_a {
                            (&host_buf_a, &dev_in_a, event_copy_done_a)
                        } else {
                            (&host_buf_b, &dev_in_b, event_copy_done_b)
                        };

                        // H2D copy on copy stream
                        {
                            crate::profile_scope!("GPU::H2D_Copy");
                            cudaMemcpyAsync(
                                *curr_dev.device_ptr() as *mut c_void,
                                curr_host.ptr() as *const c_void,
                                current_read_len * std::mem::size_of::<f64>(),
                                CUDA_MEMCPY_HOST_TO_DEVICE,
                                stream_copy.stream as *mut c_void
                            );

                            // Record event when copy completes
                            cudaEventRecord(curr_event, stream_copy.stream as *mut c_void);
                        }

                        // Make compute stream wait for copy to complete
                        {
                            crate::profile_scope!("GPU::StreamWait");
                            cudaStreamWaitEvent(stream_compute.stream as *mut c_void, curr_event, 0);
                        }

                        // Launch fused kernel (norm + encode in one pass)
                        {
                            crate::profile_scope!("GPU::FusedKernel");
                            let offset_elements = global_sample_offset * state_len_per_sample;
                            let state_ptr_offset = total_state_vector.ptr().cast::<u8>()
                                .add(offset_elements * std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
                                .cast::<std::ffi::c_void>();

                            let ret = launch_fused_amplitude_encode_batch(
                                *curr_dev.device_ptr() as *const f64,
                                state_ptr_offset,
                                samples_in_chunk,
                                sample_size,
                                state_len_per_sample,
                                stream_compute.stream as *mut c_void
                            );

                            if ret != 0 {
                                return Err(MahoutError::KernelLaunch(format!("Fused kernel error: {}", ret)));
                            }
                        }

                        global_sample_offset += samples_in_chunk;
                    }

                    // Read next chunk while GPU processes current chunk
                    {
                        crate::profile_scope!("IO::ReadNext");
                        use_buffer_a = !use_buffer_a;

                        let (next_host, next_event) = if use_buffer_a {
                            (&mut host_buf_a, event_copy_done_a)
                        } else {
                            (&mut host_buf_b, event_copy_done_b)
                        };

                        // Wait for GPU copy to finish before CPU overwrites buffer
                        cudaEventSynchronize(next_event);

                        // Read next chunk (overlaps with GPU processing)
                        current_read_len = reader.read_chunk(next_host.as_slice_mut())?;
                    }
                }
            }

            self.device.synchronize().map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;

            // Clean up events (streams auto-destroyed by cudarc Drop)
            unsafe {
                cudaEventDestroy(event_copy_done_a);
                cudaEventDestroy(event_copy_done_b);
            }

            // Transfer ownership to DLPack deleter (prevent double-free)
            let dlpack_ptr = total_state_vector.to_dlpack();
            std::mem::forget(total_state_vector);

            Ok(dlpack_ptr)
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
