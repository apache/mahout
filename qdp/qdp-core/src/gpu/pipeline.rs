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

use std::sync::Arc;
use std::ffi::c_void;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, safe::CudaStream};
use crate::error::{MahoutError, Result};
#[cfg(target_os = "linux")]
use crate::gpu::memory::{ensure_device_memory_available, map_allocation_error};
#[cfg(target_os = "linux")]
use crate::gpu::buffer_pool::{PinnedBufferPool, PinnedBufferHandle};
#[cfg(target_os = "linux")]
use crate::gpu::cuda_ffi::{
    cudaEventCreateWithFlags,
    cudaEventDestroy,
    cudaEventRecord,
    cudaMemcpyAsync,
    cudaStreamSynchronize,
    cudaStreamWaitEvent,
    CUDA_EVENT_DISABLE_TIMING,
    CUDA_MEMCPY_HOST_TO_DEVICE,
};

/// Dual-stream context coordinating copy/compute with an event.
#[cfg(target_os = "linux")]
pub struct PipelineContext {
    pub stream_compute: CudaStream,
    pub stream_copy: CudaStream,
    events_copy_done: Vec<*mut c_void>,
}

#[cfg(target_os = "linux")]
#[allow(unsafe_op_in_unsafe_fn)]
impl PipelineContext {
    pub fn new(device: &Arc<CudaDevice>, event_slots: usize) -> Result<Self> {
        let stream_compute = device.fork_default_stream()
            .map_err(|e| MahoutError::Cuda(format!("Failed to create compute stream: {:?}", e)))?;
        let stream_copy = device.fork_default_stream()
            .map_err(|e| MahoutError::Cuda(format!("Failed to create copy stream: {:?}", e)))?;

        let mut events_copy_done = Vec::with_capacity(event_slots);
        for _ in 0..event_slots {
            let mut ev: *mut c_void = std::ptr::null_mut();
            unsafe {
                let ret = cudaEventCreateWithFlags(&mut ev, CUDA_EVENT_DISABLE_TIMING);
                if ret != 0 {
                    return Err(MahoutError::Cuda(format!("Failed to create CUDA event: {}", ret)));
                }
            }
            events_copy_done.push(ev);
        }

        Ok(Self {
            stream_compute,
            stream_copy,
            events_copy_done,
        })
    }

    /// Async H2D copy on the copy stream.
    pub unsafe fn async_copy_to_device(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        len_elements: usize,
    ) -> Result<()> {
        crate::profile_scope!("GPU::H2D_Copy");
        let ret = cudaMemcpyAsync(
            dst,
            src,
            len_elements * std::mem::size_of::<f64>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
            self.stream_copy.stream as *mut c_void,
        );
        if ret != 0 {
            return Err(MahoutError::Cuda(format!("Async H2D copy failed with CUDA error: {}", ret)));
        }
        Ok(())
    }

    /// Record completion of the copy on the copy stream.
    pub unsafe fn record_copy_done(&self, slot: usize) -> Result<()> {
        let ret = cudaEventRecord(self.events_copy_done[slot], self.stream_copy.stream as *mut c_void);
        if ret != 0 {
            return Err(MahoutError::Cuda(format!("cudaEventRecord failed: {}", ret)));
        }
        Ok(())
    }

    /// Make compute stream wait for the copy completion event.
    pub unsafe fn wait_for_copy(&self, slot: usize) -> Result<()> {
        crate::profile_scope!("GPU::StreamWait");
        let ret = cudaStreamWaitEvent(self.stream_compute.stream as *mut c_void, self.events_copy_done[slot], 0);
        if ret != 0 {
            return Err(MahoutError::Cuda(format!("cudaStreamWaitEvent failed: {}", ret)));
        }
        Ok(())
    }

    /// Sync copy stream (safe to reuse host buffer).
    pub unsafe fn sync_copy_stream(&self) -> Result<()> {
        crate::profile_scope!("Pipeline::SyncCopy");
        let ret = cudaStreamSynchronize(self.stream_copy.stream as *mut c_void);
        if ret != 0 {
            return Err(MahoutError::Cuda(format!("cudaStreamSynchronize(copy) failed: {}", ret)));
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl Drop for PipelineContext {
    fn drop(&mut self) {
        unsafe {
            for ev in &mut self.events_copy_done {
                if !ev.is_null() {
                    let _ = cudaEventDestroy(*ev);
                }
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

    // Pinned host staging pool sized to the current chunking strategy (double-buffer by default).
    const CHUNK_SIZE_ELEMENTS: usize = 8 * 1024 * 1024 / std::mem::size_of::<f64>(); // 8MB
    const PINNED_POOL_SIZE: usize = 2; // double buffering
    // 1. Create dual streams with per-slot events to coordinate copy -> compute
    let ctx = PipelineContext::new(device, PINNED_POOL_SIZE)?;
    let pinned_pool = PinnedBufferPool::new(PINNED_POOL_SIZE, CHUNK_SIZE_ELEMENTS)
        .map_err(|e| MahoutError::Cuda(format!("Failed to create pinned buffer pool: {}", e)))?;

    // 2. Chunk size: 8MB per chunk (balance between overhead and overlap opportunity)
    // TODO: tune dynamically based on GPU/PCIe bandwidth.

    // 3. Keep temporary buffers alive until all streams complete
    // This prevents Rust from dropping them while GPU is still using them
    let mut keep_alive_buffers: Vec<CudaSlice<f64>> = Vec::new();
    // Keep pinned buffers alive until the copy stream has completed their H2D copy
    let mut in_flight_pinned: Vec<PinnedBufferHandle> = Vec::new();

    let mut global_offset = 0;
    let mut chunk_idx = 0usize;

    // 4. Pipeline loop: copy on copy stream, compute on compute stream with event handoff
    for chunk in host_data.chunks(CHUNK_SIZE_ELEMENTS) {
        let chunk_offset = global_offset;
        let event_slot = chunk_idx % PINNED_POOL_SIZE;

        crate::profile_scope!("GPU::ChunkProcess");

        if chunk.len() > CHUNK_SIZE_ELEMENTS {
            return Err(MahoutError::InvalidInput(format!(
                "Chunk size {} exceeds pinned buffer capacity {}",
                chunk.len(),
                CHUNK_SIZE_ELEMENTS
            )));
        }

        let chunk_bytes = chunk.len() * std::mem::size_of::<f64>();
        ensure_device_memory_available(chunk_bytes, "pipeline chunk buffer allocation", None)?;

        // Allocate temporary device buffer for this chunk
        let input_chunk_dev = unsafe {
            device.alloc::<f64>(chunk.len())
        }.map_err(|e| map_allocation_error(
            chunk_bytes,
            "pipeline chunk buffer allocation",
            None,
            e,
        ))?;

        // Acquire pinned staging buffer and populate it with the current chunk
        let mut pinned_buf = pinned_pool.acquire();
        pinned_buf.as_slice_mut()[..chunk.len()].copy_from_slice(chunk);

        // Async copy: host to device (non-blocking, on specified stream)
        // Uses CUDA Runtime API (cudaMemcpyAsync) for true async copy
        {
            crate::profile_scope!("GPU::H2DCopyAsync");
            unsafe {
                ctx.async_copy_to_device(
                    pinned_buf.ptr() as *const c_void,
                    *input_chunk_dev.device_ptr() as *mut c_void,
                    chunk.len(),
                )?;
                ctx.record_copy_done(event_slot)?;
                ctx.wait_for_copy(event_slot)?;
            }
        }

        // Keep pinned buffer alive until the copy stream is synchronized.
        in_flight_pinned.push(pinned_buf);
        if in_flight_pinned.len() == PINNED_POOL_SIZE {
            // Ensure previous H2D copies are done before reusing buffers.
            unsafe { ctx.sync_copy_stream()?; }
            in_flight_pinned.clear();
        }

        // Get device pointer for kernel launch
        let input_ptr = *input_chunk_dev.device_ptr() as *const f64;

        // Invoke caller's kernel launcher (non-blocking)
        {
            crate::profile_scope!("GPU::KernelLaunchAsync");
            kernel_launcher(&ctx.stream_compute, input_ptr, chunk_offset, chunk.len())?;
        }

        // Keep buffer alive until synchronization
        // Critical: Rust will drop CudaSlice when it goes out of scope, which calls cudaFree.
        // We must keep these buffers alive until all GPU work completes.
        keep_alive_buffers.push(input_chunk_dev);

        // Update offset for next chunk
        global_offset += chunk.len();
        chunk_idx += 1;
    }

    // 5. Synchronize all streams: wait for all work to complete
    // This ensures all async copies and kernel launches have finished
    {
        crate::profile_scope!("GPU::StreamSync");
        unsafe { ctx.sync_copy_stream()?; }
        device.wait_for(&ctx.stream_compute)
            .map_err(|e| MahoutError::Cuda(format!("Compute stream sync failed: {:?}", e)))?;
    }

    // Buffers are dropped here (after sync), freeing GPU memory
    // This is safe because all GPU operations have completed
    drop(keep_alive_buffers);

    Ok(())
}
