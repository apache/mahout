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

// Async Pipeline Infrastructure
//
// Provides generic double-buffered execution for large data processing.
// Separates the "streaming mechanics" from the "kernel logic".

use crate::error::{MahoutError, Result};
#[cfg(target_os = "linux")]
use crate::gpu::memory::{PinnedBuffer, ensure_device_memory_available, map_allocation_error};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, safe::CudaStream};
use std::ffi::c_void;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::gpu::cuda_ffi::{
    CUDA_EVENT_DISABLE_TIMING, CUDA_MEMCPY_HOST_TO_DEVICE, cudaEventCreateWithFlags,
    cudaEventDestroy, cudaEventRecord, cudaMemcpyAsync, cudaStreamSynchronize, cudaStreamWaitEvent,
};

/// Dual-stream pipeline context: manages compute/copy streams and sync events
#[cfg(target_os = "linux")]
pub struct PipelineContext {
    pub stream_compute: CudaStream,
    pub stream_copy: CudaStream,
    event_copy_done: *mut c_void,
}

#[cfg(target_os = "linux")]
impl PipelineContext {
    /// Create dual streams and sync event
    pub fn new(device: &Arc<CudaDevice>) -> Result<Self> {
        let stream_compute = device
            .fork_default_stream()
            .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
        let stream_copy = device
            .fork_default_stream()
            .map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;

        let mut event_copy_done: *mut c_void = std::ptr::null_mut();
        unsafe {
            let ret = cudaEventCreateWithFlags(&mut event_copy_done, CUDA_EVENT_DISABLE_TIMING);
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "Failed to create CUDA event: {}",
                    ret
                )));
            }
        }

        Ok(Self {
            stream_compute,
            stream_copy,
            event_copy_done,
        })
    }

    /// Async H2D copy on copy stream
    pub unsafe fn async_copy_to_device(
        &self,
        src: &PinnedBuffer,
        dst: *mut c_void,
        len_elements: usize,
    ) {
        crate::profile_scope!("GPU::H2D_Copy");
        unsafe {
            cudaMemcpyAsync(
                dst,
                src.ptr() as *const c_void,
                len_elements * std::mem::size_of::<f64>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream_copy.stream as *mut c_void,
            );
        }
    }

    /// Record copy completion event
    pub unsafe fn record_copy_done(&self) {
        unsafe {
            cudaEventRecord(self.event_copy_done, self.stream_copy.stream as *mut c_void);
        }
    }

    /// Make compute stream wait for copy completion
    pub unsafe fn wait_for_copy(&self) {
        crate::profile_scope!("GPU::StreamWait");
        unsafe {
            cudaStreamWaitEvent(
                self.stream_compute.stream as *mut c_void,
                self.event_copy_done,
                0,
            );
        }
    }

    /// Sync copy stream (safe to reuse host buffer)
    pub unsafe fn sync_copy_stream(&self) {
        crate::profile_scope!("Pipeline::SyncCopy");
        unsafe {
            cudaStreamSynchronize(self.stream_copy.stream as *mut c_void);
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for PipelineContext {
    fn drop(&mut self) {
        unsafe {
            if !self.event_copy_done.is_null() {
                cudaEventDestroy(self.event_copy_done);
            }
        }
    }
}

/// Chunk processing callback for async pipeline
///
/// This closure is called for each chunk with:
/// - `stream`: The CUDA stream to launch the kernel on
/// - `input_ptr`: Device pointer to the chunk data (already copied)
/// - `chunk_offset`: Global offset in the original data (in elements)
/// - `chunk_len`: Length of this chunk (in elements)
pub type ChunkProcessor = dyn FnMut(&CudaStream, *const f64, usize, usize) -> Result<()>;

/// Executes a task using dual-stream double-buffering pattern
///
/// This function handles the generic pipeline mechanics:
/// - Dual stream creation and management
/// - Data chunking and async H2D copy
/// - Buffer lifetime management
/// - Stream synchronization
///
/// The caller provides a `kernel_launcher` closure that handles the
/// specific kernel launch logic for each chunk.
///
/// # Arguments
/// * `device` - The CUDA device
/// * `host_data` - Full source data to process
/// * `kernel_launcher` - Closure that launches the specific kernel for each chunk
///
/// # Example
/// ```rust,ignore
/// run_dual_stream_pipeline(device, host_data, |stream, input_ptr, offset, len| {
///     // Launch your specific kernel here
///     launch_my_kernel(input_ptr, offset, len, stream)?;
///     Ok(())
/// })?;
/// ```
#[cfg(target_os = "linux")]
pub fn run_dual_stream_pipeline<F>(
    device: &Arc<CudaDevice>,
    host_data: &[f64],
    mut kernel_launcher: F,
) -> Result<()>
where
    F: FnMut(&CudaStream, *const f64, usize, usize) -> Result<()>,
{
    crate::profile_scope!("GPU::AsyncPipeline");

    // 1. Create dual streams for pipeline overlap
    let stream1 = device
        .fork_default_stream()
        .map_err(|e| MahoutError::Cuda(format!("Failed to create stream 1: {:?}", e)))?;
    let stream2 = device
        .fork_default_stream()
        .map_err(|e| MahoutError::Cuda(format!("Failed to create stream 2: {:?}", e)))?;
    let streams = [&stream1, &stream2];

    // Pin the full host buffer once so cudaMemcpyAsync can leverage DMA without staging.
    let pinned_host = PinnedBuffer::register(host_data)
        .map_err(|e| MahoutError::Cuda(format!("Failed to pin host input for async pipeline: {}", e)))?;

    // 2. Chunk size: 8MB per chunk (balance between overhead and overlap opportunity)
    // TODO: we should tune this dynamically based on the detected GPU model or PCIe bandwidth in the future.
    // Too small = launch overhead dominates, too large = less overlap
    const CHUNK_SIZE_ELEMENTS: usize = 8 * 1024 * 1024 / std::mem::size_of::<f64>(); // 8MB

    // 3. Keep temporary buffers alive until all streams complete
    // This prevents Rust from dropping them while GPU is still using them
    let mut keep_alive_buffers: Vec<CudaSlice<f64>> = Vec::new();

    let mut global_offset = 0;

    // 4. Pipeline loop: alternate between streams for maximum overlap
    for (chunk_idx, chunk) in host_data.chunks(CHUNK_SIZE_ELEMENTS).enumerate() {
        let chunk_offset = global_offset;
        let current_stream = streams[chunk_idx % 2];

        crate::profile_scope!("GPU::ChunkProcess");

        let chunk_bytes = chunk.len() * std::mem::size_of::<f64>();
        ensure_device_memory_available(chunk_bytes, "pipeline chunk buffer allocation", None)?;

        // Allocate temporary device buffer for this chunk
        let input_chunk_dev = unsafe { device.alloc::<f64>(chunk.len()) }.map_err(|e| {
            map_allocation_error(chunk_bytes, "pipeline chunk buffer allocation", None, e)
        })?;

        // Async copy: host to device (non-blocking, on specified stream)
        // Uses CUDA Runtime API (cudaMemcpyAsync) for true async copy
        {
            crate::profile_scope!("GPU::H2DCopyAsync");
            unsafe {
                let dst_device_ptr = *input_chunk_dev.device_ptr() as *mut c_void;
                let src_host_ptr = pinned_host.as_ptr().add(chunk_offset) as *const c_void;
                let bytes = chunk.len() * std::mem::size_of::<f64>();
                let stream_handle = current_stream.stream as *mut c_void;

                let result = cudaMemcpyAsync(
                    dst_device_ptr,
                    src_host_ptr,
                    bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream_handle,
                );

                if result != 0 {
                    return Err(MahoutError::Cuda(format!(
                        "Async H2D copy failed with CUDA error: {}",
                        result
                    )));
                }
            }
        }

        // Get device pointer for kernel launch
        let input_ptr = *input_chunk_dev.device_ptr() as *const f64;

        // Invoke caller's kernel launcher (non-blocking)
        {
            crate::profile_scope!("GPU::KernelLaunchAsync");
            kernel_launcher(current_stream, input_ptr, chunk_offset, chunk.len())?;
        }

        // Keep buffer alive until synchronization
        // Critical: Rust will drop CudaSlice when it goes out of scope, which calls cudaFree.
        // We must keep these buffers alive until all GPU work completes.
        keep_alive_buffers.push(input_chunk_dev);

        // Update offset for next chunk
        global_offset += chunk.len();
    }

    // 5. Synchronize all streams: wait for all work to complete
    // This ensures all async copies and kernel launches have finished
    {
        crate::profile_scope!("GPU::StreamSync");
        device
            .wait_for(&stream1)
            .map_err(|e| MahoutError::Cuda(format!("Stream 1 sync failed: {:?}", e)))?;
        device
            .wait_for(&stream2)
            .map_err(|e| MahoutError::Cuda(format!("Stream 2 sync failed: {:?}", e)))?;
    }

    // Buffers are dropped here (after sync), freeing GPU memory
    // This is safe because all GPU operations have completed
    drop(keep_alive_buffers);

    Ok(())
}
