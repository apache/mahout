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

// Allow unused_unsafe: CUDA FFI functions are unsafe in CUDA builds but safe stubs in no-CUDA builds.
// The compiler can't statically determine which path is taken.
#![allow(unused_unsafe)]

use crate::error::{MahoutError, Result};
#[cfg(target_os = "linux")]
use crate::gpu::buffer_pool::{PinnedBufferHandle, PinnedBufferPool};
#[cfg(target_os = "linux")]
use crate::gpu::cuda_ffi::{
    CUDA_EVENT_DISABLE_TIMING, CUDA_MEMCPY_HOST_TO_DEVICE, cudaEventCreateWithFlags,
    cudaEventDestroy, cudaEventRecord, cudaMemcpyAsync, cudaStreamSynchronize, cudaStreamWaitEvent,
};
#[cfg(target_os = "linux")]
use crate::gpu::memory::{ensure_device_memory_available, map_allocation_error};
#[cfg(target_os = "linux")]
use crate::gpu::overlap_tracker::OverlapTracker;
#[cfg(target_os = "linux")]
use crate::gpu::pool_metrics::PoolMetrics;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, safe::CudaStream};
use std::ffi::c_void;
use std::sync::Arc;

/// Dual-stream context coordinating copy/compute with an event.
#[cfg(target_os = "linux")]
pub struct PipelineContext {
    pub stream_compute: CudaStream,
    pub stream_copy: CudaStream,
    events_copy_done: Vec<*mut c_void>,
}

#[cfg(target_os = "linux")]
fn validate_event_slot(events: &[*mut c_void], slot: usize) -> Result<()> {
    if slot >= events.len() {
        return Err(MahoutError::InvalidInput(format!(
            "Event slot {} out of range (max: {})",
            slot,
            events.len().saturating_sub(1)
        )));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
impl PipelineContext {
    pub fn new(device: &Arc<CudaDevice>, event_slots: usize) -> Result<Self> {
        let stream_compute = device
            .fork_default_stream()
            .map_err(|e| MahoutError::Cuda(format!("Failed to create compute stream: {:?}", e)))?;
        let stream_copy = device
            .fork_default_stream()
            .map_err(|e| MahoutError::Cuda(format!("Failed to create copy stream: {:?}", e)))?;

        let mut events_copy_done = Vec::with_capacity(event_slots);
        for _ in 0..event_slots {
            let mut ev: *mut c_void = std::ptr::null_mut();
            unsafe {
                let ret = cudaEventCreateWithFlags(&mut ev, CUDA_EVENT_DISABLE_TIMING);
                if ret != 0 {
                    return Err(MahoutError::Cuda(format!(
                        "Failed to create CUDA event: {}",
                        ret
                    )));
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
    ///
    /// # Safety
    /// `src` must be valid for `len_elements` `f64` values and properly aligned.
    /// `dst` must point to device memory for `len_elements` `f64` values on the same device.
    /// Both pointers must remain valid until the copy completes on `stream_copy`.
    pub unsafe fn async_copy_to_device(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        len_elements: usize,
    ) -> Result<()> {
        crate::profile_scope!("GPU::H2D_Copy");
        unsafe {
            let ret = cudaMemcpyAsync(
                dst,
                src,
                len_elements * std::mem::size_of::<f64>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream_copy.stream as *mut c_void,
            );
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "Async H2D copy failed with CUDA error: {}",
                    ret
                )));
            }
        }
        Ok(())
    }

    /// Record completion of the copy on the copy stream.
    ///
    /// # Safety
    /// `slot` must refer to a live event created by this context, and the context must
    /// remain alive until the event is no longer used by any stream.
    pub unsafe fn record_copy_done(&self, slot: usize) -> Result<()> {
        validate_event_slot(&self.events_copy_done, slot)?;

        unsafe {
            let ret = cudaEventRecord(
                self.events_copy_done[slot],
                self.stream_copy.stream as *mut c_void,
            );
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventRecord failed: {}",
                    ret
                )));
            }
        }
        Ok(())
    }

    /// Make compute stream wait for the copy completion event.
    ///
    /// # Safety
    /// `slot` must refer to a live event previously recorded on `stream_copy`, and the
    /// context and its streams must remain valid while waiting.
    pub unsafe fn wait_for_copy(&self, slot: usize) -> Result<()> {
        crate::profile_scope!("GPU::StreamWait");
        validate_event_slot(&self.events_copy_done, slot)?;

        unsafe {
            let ret = cudaStreamWaitEvent(
                self.stream_compute.stream as *mut c_void,
                self.events_copy_done[slot],
                0,
            );
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "cudaStreamWaitEvent failed: {}",
                    ret
                )));
            }
        }
        Ok(())
    }

    /// Sync copy stream (safe to reuse host buffer).
    ///
    /// # Safety
    /// The context and its copy stream must be valid and not destroyed while syncing.
    pub unsafe fn sync_copy_stream(&self) -> Result<()> {
        crate::profile_scope!("Pipeline::SyncCopy");
        unsafe {
            let ret = cudaStreamSynchronize(self.stream_copy.stream as *mut c_void);
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "cudaStreamSynchronize(copy) failed: {}",
                    ret
                )));
            }
        }
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::validate_event_slot;

    #[test]
    fn validate_event_slot_allows_in_range() {
        let events = vec![std::ptr::null_mut(); 2];
        assert!(validate_event_slot(&events, 0).is_ok());
        assert!(validate_event_slot(&events, 1).is_ok());
    }

    #[test]
    fn validate_event_slot_rejects_out_of_range() {
        let events = vec![std::ptr::null_mut(); 2];
        let err = validate_event_slot(&events, 2).unwrap_err();
        assert!(matches!(err, crate::error::MahoutError::InvalidInput(_)));
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

/// Executes a task using dual-stream double-buffering pattern
///
/// This function handles the generic pipeline mechanics:
/// - Dual stream creation and management
/// - Data chunking and async H2D copy
/// - Buffer lifetime management
/// - Stream synchronization
/// - Optional observability (pool metrics and overlap tracking)
///
/// The caller provides a `kernel_launcher` closure that handles the
/// specific kernel launch logic for each chunk.
///
/// # Arguments
/// * `device` - The CUDA device
/// * `host_data` - Full source data to process
/// * `kernel_launcher` - Closure that launches the specific kernel for each chunk
///
/// # Environment Variables
/// * `QDP_ENABLE_POOL_METRICS` - Enable pool utilization metrics (set to "1" or "true")
/// * `QDP_ENABLE_OVERLAP_TRACKING` - Enable H2D overlap tracking (set to "1" or "true")
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

    // Check environment variables for observability features
    let enable_pool_metrics = std::env::var("QDP_ENABLE_POOL_METRICS")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let enable_overlap_tracking = std::env::var("QDP_ENABLE_OVERLAP_TRACKING")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // 1. Create dual streams with per-slot events to coordinate copy -> compute
    let ctx = PipelineContext::new(device, PINNED_POOL_SIZE)?;
    let pinned_pool = PinnedBufferPool::new(PINNED_POOL_SIZE, CHUNK_SIZE_ELEMENTS)
        .map_err(|e| MahoutError::Cuda(format!("Failed to create pinned buffer pool: {}", e)))?;

    // Initialize observability tools (optional)
    let pool_metrics = if enable_pool_metrics {
        Some(Arc::new(PoolMetrics::new()))
    } else {
        None
    };
    let overlap_tracker = if enable_overlap_tracking {
        Some(OverlapTracker::new(PINNED_POOL_SIZE, true)?)
    } else {
        None
    };

    // 2. Chunk size: 8MB per chunk (balance between overhead and overlap opportunity)
    // TODO: tune dynamically based on GPU/PCIe bandwidth.

    // 3. Keep temporary buffers alive until all streams complete
    // This prevents Rust from dropping them while GPU is still using them
    let mut keep_alive_buffers: Vec<CudaSlice<f64>> = Vec::new();
    // Keep pinned buffers alive until the copy stream has completed their H2D copy
    let mut in_flight_pinned: Vec<PinnedBufferHandle> = Vec::new();

    let mut global_offset = 0;

    // 4. Pipeline loop: copy on copy stream, compute on compute stream with event handoff
    for (chunk_idx, chunk) in host_data.chunks(CHUNK_SIZE_ELEMENTS).enumerate() {
        let chunk_offset = global_offset;
        let event_slot = chunk_idx % PINNED_POOL_SIZE;

        crate::profile_scope!("GPU::ChunkProcess");

        let chunk_bytes = std::mem::size_of_val(chunk);
        ensure_device_memory_available(chunk_bytes, "pipeline chunk buffer allocation", None)?;

        // Allocate temporary device buffer for this chunk
        let input_chunk_dev = unsafe { device.alloc::<f64>(chunk.len()) }.map_err(|e| {
            map_allocation_error(chunk_bytes, "pipeline chunk buffer allocation", None, e)
        })?;

        // Acquire pinned staging buffer and populate it with the current chunk
        let mut pinned_buf = if let Some(ref metrics) = pool_metrics {
            pinned_pool.acquire_with_metrics(Some(metrics.as_ref()))
        } else {
            pinned_pool.acquire()
        };
        pinned_buf.as_slice_mut()[..chunk.len()].copy_from_slice(chunk);

        // Async copy: host to device (non-blocking, on specified stream)
        // Uses CUDA Runtime API (cudaMemcpyAsync) for true async copy
        {
            crate::profile_scope!("GPU::H2DCopyAsync");

            // Record copy start if overlap tracking enabled
            // Note: Overlap tracking is optional observability - failures should not stop the pipeline
            if let Some(ref tracker) = overlap_tracker
                && let Err(e) = tracker.record_copy_start(&ctx.stream_copy, event_slot)
            {
                log::warn!(
                    "Chunk {}: Failed to record copy start event: {}. Overlap tracking may be incomplete.",
                    chunk_idx,
                    e
                );
            }

            unsafe {
                ctx.async_copy_to_device(
                    pinned_buf.ptr() as *const c_void,
                    *input_chunk_dev.device_ptr() as *mut c_void,
                    chunk.len(),
                )?;

                // Record copy end if overlap tracking enabled
                // Note: Overlap tracking is optional observability - failures should not stop the pipeline
                if let Some(ref tracker) = overlap_tracker
                    && let Err(e) = tracker.record_copy_end(&ctx.stream_copy, event_slot)
                {
                    log::warn!(
                        "Chunk {}: Failed to record copy end event: {}. Overlap tracking may be incomplete.",
                        chunk_idx,
                        e
                    );
                }

                ctx.record_copy_done(event_slot)?;
                ctx.wait_for_copy(event_slot)?;
            }
        }

        // Keep pinned buffer alive until the copy stream is synchronized.
        in_flight_pinned.push(pinned_buf);
        if in_flight_pinned.len() == PINNED_POOL_SIZE {
            // Ensure previous H2D copies are done before reusing buffers.
            unsafe {
                ctx.sync_copy_stream()?;
            }
            in_flight_pinned.clear();
        }

        // Get device pointer for kernel launch
        let input_ptr = *input_chunk_dev.device_ptr() as *const f64;

        // Invoke caller's kernel launcher (non-blocking)
        {
            crate::profile_scope!("GPU::KernelLaunchAsync");

            // Record compute start if overlap tracking enabled
            // Note: Overlap tracking is optional observability - failures should not stop the pipeline
            if let Some(ref tracker) = overlap_tracker
                && let Err(e) = tracker.record_compute_start(&ctx.stream_compute, event_slot)
            {
                log::warn!(
                    "Chunk {}: Failed to record compute start event: {}. Overlap tracking may be incomplete.",
                    chunk_idx,
                    e
                );
            }

            kernel_launcher(&ctx.stream_compute, input_ptr, chunk_offset, chunk.len())?;

            // Record compute end if overlap tracking enabled
            // Note: Overlap tracking is optional observability - failures should not stop the pipeline
            if let Some(ref tracker) = overlap_tracker
                && let Err(e) = tracker.record_compute_end(&ctx.stream_compute, event_slot)
            {
                log::warn!(
                    "Chunk {}: Failed to record compute end event: {}. Overlap tracking may be incomplete.",
                    chunk_idx,
                    e
                );
            }
        }

        // Log overlap if tracking enabled
        // Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        // We log after recording events. log_overlap will wait for events to complete
        // before calculating elapsed time. This ensures accurate measurements.
        //
        // Note: log_overlap now handles both success and failure cases internally,
        // logging at appropriate levels (INFO for visibility, DEBUG for details).
        if let Some(ref tracker) = overlap_tracker
            && (chunk_idx % 10 == 0 || chunk_idx == 0)
        {
            // Only log every Nth chunk to avoid excessive logging
            // Note: log_overlap waits for events to complete, which may take time
            // If events fail (e.g., invalid resource handle), log_overlap will log
            // at INFO level so it's visible in both debug and info modes
            if let Err(e) = tracker.log_overlap(chunk_idx) {
                // log_overlap already logged the error at INFO level
                // We only need to log additional details at DEBUG level if needed
                if log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "Overlap tracking failed for chunk {}: {}. Pipeline continues normally.",
                        chunk_idx,
                        e
                    );
                }
                // Don't fail the pipeline - overlap tracking is optional observability
            }
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
        unsafe {
            ctx.sync_copy_stream()?;
        }
        device
            .wait_for(&ctx.stream_compute)
            .map_err(|e| MahoutError::Cuda(format!("Compute stream sync failed: {:?}", e)))?;
    }

    // Buffers are dropped here (after sync), freeing GPU memory
    // This is safe because all GPU operations have completed
    drop(keep_alive_buffers);

    // Print pool metrics summary if enabled
    if let Some(ref metrics) = pool_metrics {
        let report = metrics.report();
        report.print_summary();
    }

    Ok(())
}
