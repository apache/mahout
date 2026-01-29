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

//! Pool utilization metrics for diagnosing pool starvation.
//!
//! Provides lightweight, thread-safe metrics tracking for pinned buffer pool
//! utilization. Uses lock-free atomic operations to minimize performance impact.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Pool utilization metrics (thread-safe, lock-free design)
///
/// Tracks pool availability, acquire operations, and wait times to diagnose
/// pool starvation issues. Uses atomic operations with Relaxed ordering for
/// minimal performance overhead.
#[derive(Debug)]
pub struct PoolMetrics {
    min_available: AtomicUsize,
    max_available: AtomicUsize,
    total_acquires: AtomicU64,
    total_waits: AtomicU64, // Number of times pool was empty when acquiring
    total_wait_time_ns: AtomicU64, // Total wait time in nanoseconds
}

impl PoolMetrics {
    /// Create a new PoolMetrics instance with all counters initialized to zero.
    pub fn new() -> Self {
        Self {
            min_available: AtomicUsize::new(usize::MAX),
            max_available: AtomicUsize::new(0),
            total_acquires: AtomicU64::new(0),
            total_waits: AtomicU64::new(0),
            total_wait_time_ns: AtomicU64::new(0),
        }
    }

    /// Record an acquire operation with the number of available buffers at that time.
    ///
    /// Uses compare-and-swap loops to ensure atomicity of min/max updates
    /// and avoid race conditions under concurrent access.
    pub fn record_acquire(&self, available: usize) {
        // Update minimum available using a compare-and-swap loop to avoid races
        loop {
            let current_min = self.min_available.load(Ordering::Relaxed);
            if available >= current_min {
                break; // Current value is already <= available, no update needed
            }
            match self.min_available.compare_exchange_weak(
                current_min,
                available,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,     // Successfully updated
                Err(_) => continue, // Value changed, retry
            }
        }

        // Update maximum available using a compare-and-swap loop to avoid races
        loop {
            let current_max = self.max_available.load(Ordering::Relaxed);
            if available <= current_max {
                break; // Current value is already >= available, no update needed
            }
            match self.max_available.compare_exchange_weak(
                current_max,
                available,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,     // Successfully updated
                Err(_) => continue, // Value changed, retry
            }
        }

        self.total_acquires.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a wait operation with the wait time in nanoseconds.
    pub fn record_wait(&self, wait_time_ns: u64) {
        self.total_waits.fetch_add(1, Ordering::Relaxed);
        self.total_wait_time_ns
            .fetch_add(wait_time_ns, Ordering::Relaxed);
    }

    /// Generate a utilization report from the current metrics.
    pub fn report(&self) -> PoolUtilizationReport {
        let acquires = self.total_acquires.load(Ordering::Relaxed);
        let waits = self.total_waits.load(Ordering::Relaxed);
        let wait_time_ns = self.total_wait_time_ns.load(Ordering::Relaxed);

        PoolUtilizationReport {
            min_available: self.min_available.load(Ordering::Relaxed),
            max_available: self.max_available.load(Ordering::Relaxed),
            total_acquires: acquires,
            total_waits: waits,
            starvation_ratio: if acquires > 0 {
                waits as f64 / acquires as f64
            } else {
                0.0
            },
            avg_wait_time_ns: if waits > 0 { wait_time_ns / waits } else { 0 },
        }
    }

    /// Reset all metrics to their initial state.
    pub fn reset(&self) {
        self.min_available.store(usize::MAX, Ordering::Relaxed);
        self.max_available.store(0, Ordering::Relaxed);
        self.total_acquires.store(0, Ordering::Relaxed);
        self.total_waits.store(0, Ordering::Relaxed);
        self.total_wait_time_ns.store(0, Ordering::Relaxed);
    }
}

impl Default for PoolMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Pool utilization report containing aggregated metrics.
#[derive(Debug, Clone)]
pub struct PoolUtilizationReport {
    /// Minimum number of buffers available during any acquire operation
    pub min_available: usize,
    /// Maximum number of buffers available during any acquire operation
    pub max_available: usize,
    /// Total number of acquire operations
    pub total_acquires: u64,
    /// Total number of wait operations (pool was empty)
    pub total_waits: u64,
    /// Ratio of waits to acquires (starvation ratio)
    pub starvation_ratio: f64,
    /// Average wait time in nanoseconds
    pub avg_wait_time_ns: u64,
}

impl PoolUtilizationReport {
    /// Print a summary of the utilization report to the log.
    pub fn print_summary(&self) {
        log::info!(
            "Pool Utilization: min={}, max={}, acquires={}, waits={}, starvation={:.2}%",
            self.min_available,
            self.max_available,
            self.total_acquires,
            self.total_waits,
            self.starvation_ratio * 100.0
        );

        if self.starvation_ratio > 0.05 {
            log::warn!(
                "Pool starvation detected: {:.1}% of acquires had to wait. Consider increasing pool size.",
                self.starvation_ratio * 100.0
            );
        }

        if self.avg_wait_time_ns > 0 {
            let avg_wait_ms = self.avg_wait_time_ns as f64 / 1_000_000.0;
            log::info!("Average wait time: {:.3} ms", avg_wait_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_metrics_new() {
        let metrics = PoolMetrics::new();
        let report = metrics.report();
        assert_eq!(report.total_acquires, 0);
        assert_eq!(report.total_waits, 0);
        assert_eq!(report.min_available, usize::MAX);
        assert_eq!(report.max_available, 0);
    }

    #[test]
    fn test_pool_metrics_record_acquire() {
        let metrics = PoolMetrics::new();
        metrics.record_acquire(2);
        metrics.record_acquire(1);
        metrics.record_acquire(3);

        let report = metrics.report();
        assert_eq!(report.total_acquires, 3);
        assert_eq!(report.min_available, 1);
        assert_eq!(report.max_available, 3);
    }

    #[test]
    fn test_pool_metrics_record_wait() {
        let metrics = PoolMetrics::new();
        metrics.record_wait(1_000_000); // 1ms
        metrics.record_wait(2_000_000); // 2ms

        let report = metrics.report();
        assert_eq!(report.total_waits, 2);
        assert_eq!(report.avg_wait_time_ns, 1_500_000);
    }

    #[test]
    fn test_pool_metrics_starvation_ratio() {
        let metrics = PoolMetrics::new();
        metrics.record_acquire(2);
        metrics.record_acquire(1);
        metrics.record_wait(1_000_000);

        let report = metrics.report();
        assert_eq!(report.total_acquires, 2);
        assert_eq!(report.total_waits, 1);
        assert!((report.starvation_ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pool_metrics_reset() {
        let metrics = PoolMetrics::new();
        metrics.record_acquire(2);
        metrics.record_wait(1_000_000);

        metrics.reset();
        let report = metrics.report();
        assert_eq!(report.total_acquires, 0);
        assert_eq!(report.total_waits, 0);
        assert_eq!(report.min_available, usize::MAX);
        assert_eq!(report.max_available, 0);
    }
}
