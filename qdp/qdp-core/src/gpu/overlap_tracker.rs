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

//! Overlap tracking for H2D copy and compute operations.
//!
//! Uses CUDA events to measure the overlap between host-to-device copy operations
//! and compute operations, enabling verification of the >60% overlap target.

use crate::error::{MahoutError, Result};
use crate::gpu::cuda_ffi::{
    CUDA_EVENT_DEFAULT, CUDA_SUCCESS, cudaEventCreateWithFlags, cudaEventDestroy,
    cudaEventElapsedTime, cudaEventRecord, cudaEventSynchronize,
};
use cudarc::driver::safe::CudaStream;
use std::ffi::c_void;

/// Tracks overlap between H2D copy and compute operations using CUDA events.
///
/// Creates events for each pool slot to track copy and compute start/end times.
/// Can be optionally enabled via environment variable to minimize overhead when disabled.
pub struct OverlapTracker {
    copy_start_events: Vec<*mut c_void>,
    copy_end_events: Vec<*mut c_void>,
    compute_start_events: Vec<*mut c_void>,
    compute_end_events: Vec<*mut c_void>,
    pool_size: usize,
    enabled: bool,
}

impl OverlapTracker {
    /// Create a new OverlapTracker for the given pool size.
    ///
    /// If disabled, no events are created and all operations are no-ops.
    pub fn new(pool_size: usize, enabled: bool) -> Result<Self> {
        if !enabled {
            return Ok(Self {
                copy_start_events: Vec::new(),
                copy_end_events: Vec::new(),
                compute_start_events: Vec::new(),
                compute_end_events: Vec::new(),
                pool_size,
                enabled: false,
            });
        }

        let mut copy_start: Vec<*mut c_void> = Vec::with_capacity(pool_size);
        let mut copy_end: Vec<*mut c_void> = Vec::with_capacity(pool_size);
        let mut compute_start: Vec<*mut c_void> = Vec::with_capacity(pool_size);
        let mut compute_end: Vec<*mut c_void> = Vec::with_capacity(pool_size);

        unsafe {
            for _ in 0..pool_size {
                let mut ev: *mut c_void = std::ptr::null_mut();
                let ret = cudaEventCreateWithFlags(&mut ev, CUDA_EVENT_DEFAULT);
                if ret != CUDA_SUCCESS {
                    Self::cleanup_events(&[&copy_start, &copy_end, &compute_start, &compute_end]);
                    return Err(MahoutError::Cuda(format!(
                        "Failed to create CUDA event: {}",
                        ret
                    )));
                }
                copy_start.push(ev);

                let mut ev: *mut c_void = std::ptr::null_mut();
                let ret = cudaEventCreateWithFlags(&mut ev, CUDA_EVENT_DEFAULT);
                if ret != CUDA_SUCCESS {
                    Self::cleanup_events(&[&copy_start, &copy_end, &compute_start, &compute_end]);
                    return Err(MahoutError::Cuda(format!(
                        "Failed to create CUDA event: {}",
                        ret
                    )));
                }
                copy_end.push(ev);

                let mut ev: *mut c_void = std::ptr::null_mut();
                let ret = cudaEventCreateWithFlags(&mut ev, CUDA_EVENT_DEFAULT);
                if ret != CUDA_SUCCESS {
                    Self::cleanup_events(&[&copy_start, &copy_end, &compute_start, &compute_end]);
                    return Err(MahoutError::Cuda(format!(
                        "Failed to create CUDA event: {}",
                        ret
                    )));
                }
                compute_start.push(ev);

                let mut ev: *mut c_void = std::ptr::null_mut();
                let ret = cudaEventCreateWithFlags(&mut ev, CUDA_EVENT_DEFAULT);
                if ret != CUDA_SUCCESS {
                    Self::cleanup_events(&[&copy_start, &copy_end, &compute_start, &compute_end]);
                    return Err(MahoutError::Cuda(format!(
                        "Failed to create CUDA event: {}",
                        ret
                    )));
                }
                compute_end.push(ev);
            }
        }

        Ok(Self {
            copy_start_events: copy_start,
            copy_end_events: copy_end,
            compute_start_events: compute_start,
            compute_end_events: compute_end,
            pool_size,
            enabled,
        })
    }

    unsafe fn cleanup_events(events: &[&Vec<*mut c_void>]) {
        for event_vec in events {
            for ev in event_vec.iter() {
                if !ev.is_null() {
                    unsafe {
                        let _ = cudaEventDestroy(*ev);
                    }
                }
            }
        }
    }

    fn validate_slot(&self, slot: usize) -> Result<()> {
        if slot >= self.pool_size {
            return Err(MahoutError::InvalidInput(format!(
                "Slot {} out of range (max: {})",
                slot,
                self.pool_size.saturating_sub(1)
            )));
        }
        Ok(())
    }

    fn record_event(&self, event: *mut c_void, stream: &CudaStream) -> Result<()> {
        // Validate event is not null before recording
        // Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        // cudaEventRecord returns cudaErrorInvalidResourceHandle if event is NULL
        if event.is_null() {
            return Err(MahoutError::Cuda(
                "Cannot record event: event is null (invalid resource handle)".to_string(),
            ));
        }

        unsafe {
            // Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
            // cudaEventRecord captures the contents of stream at the time of this call
            // The operation is asynchronous - use cudaEventQuery or cudaEventSynchronize
            // to determine when the event has actually been recorded
            let ret = cudaEventRecord(event, stream.stream as *mut c_void);
            if ret != CUDA_SUCCESS {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventRecord failed: {} ({}). Event may be invalid or stream may be invalid.",
                    ret,
                    crate::error::cuda_error_to_string(ret)
                )));
            }
        }
        Ok(())
    }

    /// Record the start of a copy operation on the copy stream.
    pub fn record_copy_start(&self, stream: &CudaStream, slot: usize) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.validate_slot(slot)?;
        self.record_event(self.copy_start_events[slot], stream)
    }

    /// Record the end of a copy operation on the copy stream.
    pub fn record_copy_end(&self, stream: &CudaStream, slot: usize) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.validate_slot(slot)?;
        self.record_event(self.copy_end_events[slot], stream)
    }

    /// Record the start of a compute operation on the compute stream.
    pub fn record_compute_start(&self, stream: &CudaStream, slot: usize) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.validate_slot(slot)?;
        self.record_event(self.compute_start_events[slot], stream)
    }

    /// Record the end of a compute operation on the compute stream.
    pub fn record_compute_end(&self, stream: &CudaStream, slot: usize) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.validate_slot(slot)?;
        self.record_event(self.compute_end_events[slot], stream)
    }

    /// Calculate the overlap ratio for a specific chunk.
    ///
    /// Returns overlap ratio (0.0-1.0): min(copy_time, compute_time) / max(copy_time, compute_time)
    ///
    /// Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    /// Events must be recorded before querying. This function waits for events to complete.
    ///
    /// Note: For detailed timing diagnostics, use `calculate_overlap_with_timing()`.
    pub fn calculate_overlap(&self, chunk_idx: usize) -> Result<f64> {
        self.calculate_overlap_with_timing(chunk_idx)
            .map(|(overlap, _, _, _)| overlap)
    }

    /// Calculate overlap ratio with detailed timing information.
    ///
    /// Returns (overlap_ratio, copy_time_ms, compute_time_ms, overlap_time_ms)
    /// for detailed diagnostics at DEBUG level.
    fn calculate_overlap_with_timing(&self, chunk_idx: usize) -> Result<(f64, f32, f32, f32)> {
        if !self.enabled {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }

        let slot = chunk_idx % self.pool_size;

        // Validate events are not null before querying
        // Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        // cudaEventQuery returns cudaErrorInvalidResourceHandle if event is NULL
        if self.copy_end_events[slot].is_null() || self.compute_end_events[slot].is_null() {
            return Err(MahoutError::Cuda(format!(
                "Event is null for chunk {} slot {}: events may not have been created",
                chunk_idx, slot
            )));
        }

        unsafe {
            // Wait for events to complete before calculating elapsed time
            // Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
            //
            // Critical: According to CUDA docs (2026):
            // 1. cudaEventRecord() is asynchronous - the event may not be recorded immediately
            // 2. Before the first call to cudaEventRecord(), cudaEventQuery() returns cudaSuccess
            //    (because an empty event is considered "complete")
            // 3. cudaEventElapsedTime() returns cudaErrorInvalidResourceHandle (600) if either
            //    event has not been recorded with cudaEventRecord()
            //
            // Solution: Use cudaEventSynchronize() instead of cudaEventQuery() to ensure events
            // are both recorded AND completed. cudaEventSynchronize() blocks until the event
            // has been recorded and all captured work has completed.
            //
            // We synchronize end events first (they complete last), then start events.
            let ret = cudaEventSynchronize(self.copy_end_events[slot]);
            if ret != CUDA_SUCCESS {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventSynchronize (copy end) failed: {} ({}). Event may not have been recorded.",
                    ret,
                    crate::error::cuda_error_to_string(ret)
                )));
            }

            let ret = cudaEventSynchronize(self.compute_end_events[slot]);
            if ret != CUDA_SUCCESS {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventSynchronize (compute end) failed: {} ({}). Event may not have been recorded.",
                    ret,
                    crate::error::cuda_error_to_string(ret)
                )));
            }

            // Verify start events are also complete (they should complete before end events)
            let ret = cudaEventSynchronize(self.copy_start_events[slot]);
            if ret != CUDA_SUCCESS {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventSynchronize (copy start) failed: {} ({}). Event may not have been recorded.",
                    ret,
                    crate::error::cuda_error_to_string(ret)
                )));
            }

            let ret = cudaEventSynchronize(self.compute_start_events[slot]);
            if ret != CUDA_SUCCESS {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventSynchronize (compute start) failed: {} ({}). Event may not have been recorded.",
                    ret,
                    crate::error::cuda_error_to_string(ret)
                )));
            }
        }

        let mut copy_time_ms: f32 = 0.0;
        let mut compute_time_ms: f32 = 0.0;

        unsafe {
            // Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
            // cudaEventElapsedTime returns cudaErrorInvalidResourceHandle (error 600) if:
            // 1. Either event has not been recorded with cudaEventRecord()
            // 2. Either event was created with cudaEventDisableTiming flag
            // 3. Either event is NULL
            let ret = cudaEventElapsedTime(
                &mut copy_time_ms,
                self.copy_start_events[slot],
                self.copy_end_events[slot],
            );
            if ret != CUDA_SUCCESS {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventElapsedTime (copy) failed: {} ({}). Events may not have been recorded properly.",
                    ret,
                    crate::error::cuda_error_to_string(ret)
                )));
            }

            let ret = cudaEventElapsedTime(
                &mut compute_time_ms,
                self.compute_start_events[slot],
                self.compute_end_events[slot],
            );
            if ret != CUDA_SUCCESS {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventElapsedTime (compute) failed: {} ({}). Events may not have been recorded properly.",
                    ret,
                    crate::error::cuda_error_to_string(ret)
                )));
            }
        }

        let overlap_time_ms = copy_time_ms.min(compute_time_ms);
        let total_time = copy_time_ms.max(compute_time_ms);

        let overlap_ratio = if total_time > 0.0 {
            (overlap_time_ms / total_time) as f64
        } else {
            0.0
        };

        Ok((
            overlap_ratio,
            copy_time_ms,
            compute_time_ms,
            overlap_time_ms,
        ))
    }

    /// Log the overlap ratio for a specific chunk.
    ///
    /// Logs overlap percentage at INFO level (important performance metric).
    /// Logs detailed timing information at DEBUG level for troubleshooting.
    /// Ref: https://docs.rs/env_logger/latest/env_logger/
    /// - RUST_LOG=debug shows all levels (DEBUG, INFO, WARN, ERROR)
    /// - RUST_LOG=info shows INFO and above (INFO, WARN, ERROR)
    ///
    /// According to Rust logging best practices:
    /// - INFO: Useful information about normal operation (overlap percentage)
    /// - DEBUG: Lower priority information for troubleshooting (detailed timing diagnostics)
    pub fn log_overlap(&self, chunk_idx: usize) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Try to calculate overlap with detailed timing information
        match self.calculate_overlap_with_timing(chunk_idx) {
            Ok((overlap, copy_time_ms, compute_time_ms, overlap_time_ms)) => {
                // Log overlap percentage at INFO level (important performance metric)
                // Ref: Rust logging best practices - INFO for normal operation metrics
                log::info!("Chunk {}: H2D overlap = {:.1}%", chunk_idx, overlap * 100.0);

                // Log detailed timing information at DEBUG level for troubleshooting
                // Ref: Rust logging best practices - DEBUG for detailed diagnostics
                if log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "Chunk {}: H2D overlap details - copy={:.3}ms, compute={:.3}ms, overlap={:.3}ms, ratio={:.1}%",
                        chunk_idx,
                        copy_time_ms,
                        compute_time_ms,
                        overlap_time_ms,
                        overlap * 100.0
                    );
                }

                if overlap < 0.6 {
                    log::warn!(
                        "Chunk {}: Overlap below target (60%), current = {:.1}%",
                        chunk_idx,
                        overlap * 100.0
                    );
                }

                Ok(())
            }
            Err(e) => {
                // Log error at INFO level (visible in both debug and info modes)
                // Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
                log::info!(
                    "Chunk {}: H2D overlap calculation unavailable: {}. This may indicate event lifecycle issues or timing constraints.",
                    chunk_idx,
                    e
                );

                // Log additional diagnostic details at DEBUG level for troubleshooting
                if log::log_enabled!(log::Level::Debug) {
                    let slot = chunk_idx % self.pool_size;
                    log::debug!(
                        "Chunk {} (slot {}): Overlap calculation failed: {}. Event pointers: copy_end={:?}, compute_end={:?}",
                        chunk_idx,
                        slot,
                        e,
                        self.copy_end_events[slot].is_null(),
                        self.compute_end_events[slot].is_null()
                    );
                }

                // Return error so caller knows calculation failed
                Err(e)
            }
        }
    }
}

impl Drop for OverlapTracker {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        unsafe {
            Self::cleanup_events(&[
                &self.copy_start_events,
                &self.copy_end_events,
                &self.compute_start_events,
                &self.compute_end_events,
            ]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap_tracker_disabled() {
        let tracker = OverlapTracker::new(2, false).unwrap();
        assert!(!tracker.enabled);
        assert!(tracker.copy_start_events.is_empty());
        assert!(tracker.copy_end_events.is_empty());
        assert!(tracker.compute_start_events.is_empty());
        assert!(tracker.compute_end_events.is_empty());
    }
}
