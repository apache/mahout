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

//! Construction-time GPU memory guard (issue #1430).
//!
//! Three probes are in play and they are not interchangeable:
//!
//! * `common::qdp_engine_probed()` — CUDA *driver* API, inside `catch_unwind` because `cudarc`
//!   panics rather than returning `Err` when `libcuda` is absent. Answers "can an engine exist".
//! * `qdp_core::cuda_runtime_available()` — CUDA *runtime* API, stubbed out when `nvcc` is absent.
//!   Answers "will the guard actually query device memory".
//! * Neither implies the other. A `QDP_NO_CUDA` build on a machine with a driver satisfies the
//!   first and fails the second, which is exactly the configuration where the guard steps aside.

mod common;

#[cfg(target_os = "linux")]
use qdp_core::{Encoding, PipelineConfig, PipelineIterator, Precision, QdpEngine};
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

/// Amplitude, 30 qubits (the `MAX_QUBITS` ceiling), batch 64, f32 — far past any real card.
#[cfg(target_os = "linux")]
fn oversized_config() -> PipelineConfig {
    PipelineConfig {
        num_qubits: 30,
        batch_size: 64,
        total_batches: 1,
        encoding: Encoding::Amplitude,
        dtype: Precision::Float32,
        prefetch_depth: 1,
        ..PipelineConfig::default()
    }
}

/// An engine plus a working CUDA runtime, or `None` to skip.
///
/// Both are required before a test may feed the guard an oversized config: without the runtime
/// API the guard steps aside by design, and the synthetic producer would then try to allocate
/// 2^30 * 64 f32 values on its own thread and abort the test process.
#[cfg(target_os = "linux")]
fn engine_with_live_runtime() -> Option<QdpEngine> {
    let engine = common::qdp_engine_probed()?;
    if !qdp_core::cuda_runtime_available() {
        println!("SKIP: CUDA runtime unavailable (stub build)");
        return None;
    }
    Some(engine)
}

#[test]
#[cfg(target_os = "linux")]
fn oversized_config_rejected_before_allocation() {
    let Some(engine) = engine_with_live_runtime() else {
        return;
    };

    let start = Instant::now();
    let result = PipelineIterator::new_synthetic(engine, oversized_config());
    let elapsed = start.elapsed();

    let message = result
        .err()
        .expect("a 30-qubit batch-64 state buffer cannot fit in any current device")
        .to_string();

    assert!(
        message.contains("Reduce qubits or batch size"),
        "message must name a remedy that works, got: {message}"
    );
    assert!(
        !message.contains("prefetch_depth"),
        "prefetch_depth does not affect gpu_state_bytes and must not be suggested, got: {message}"
    );
    for expected in ["encoding=amplitude", "batch_size=64", "qubits=30"] {
        assert!(
            message.contains(expected),
            "message must name {expected}, got: {message}"
        );
    }

    // Pins the deliberately conservative budget: the guard compares two concurrent batch buffers,
    // not the single buffer `GpuStateVector::new_batch` allocates, because a DLPack tensor held
    // across `__next__` keeps the previous batch resident. One f32 batch buffer here is
    // 64 * 2^30 complex64 (8 bytes) = 512 GiB, so the guard must request twice that. Asserting the
    // exact figure needs no knowledge of the host's free memory.
    let single_buffer_mib = 64.0 * (1u64 << 30) as f64 * 8.0 / (1024.0 * 1024.0);
    let expected = format!("requested {:.2} MiB", 2.0 * single_buffer_mib);
    assert!(
        message.contains(&expected),
        "expected the two-buffer budget ({expected}), got: {message}"
    );

    // The point of the guard is that rejection is immediate rather than discovered mid-run.
    assert!(
        elapsed < Duration::from_secs(1),
        "rejection took {elapsed:?}, expected well under a second"
    );
}

/// The file loaders must reject before touching the filesystem. Pointing them at a path that does
/// not exist separates the two outcomes: the guard's memory error proves it ran first, while a
/// missing-file error would mean the check had been skipped or wired in too late.
#[test]
#[cfg(target_os = "linux")]
fn file_loaders_reject_before_reading_the_file() {
    let Some(engine) = engine_with_live_runtime() else {
        return;
    };
    let missing = std::path::Path::new("/nonexistent/qdp-vram-guard/never-created.parquet");

    let full_read =
        PipelineIterator::new_from_file(engine.clone(), missing, oversized_config(), usize::MAX)
            .err()
            .expect("new_from_file must reject the oversized config")
            .to_string();
    assert!(
        full_read.contains("Reduce qubits or batch size"),
        "new_from_file must fail on memory before opening the file, got: {full_read}"
    );

    let streaming =
        PipelineIterator::new_from_file_streaming(engine, missing, oversized_config(), usize::MAX)
            .err()
            .expect("new_from_file_streaming must reject the oversized config")
            .to_string();
    assert!(
        streaming.contains("Reduce qubits or batch size"),
        "new_from_file_streaming must fail on memory before opening the file, got: {streaming}"
    );
}

/// An f32 pipeline on an f64 engine holds complex128 buffers on the device, so the guard must
/// budget f64. Budgeting the config's dtype would halve the estimate and admit a config that then
/// runs out of memory mid-run. Reachable from Python: the synthetic loader hardcodes an f32 config
/// dtype while `QdpEngine(precision="float64")` is a documented option.
#[test]
#[cfg(target_os = "linux")]
fn engine_precision_widens_the_budget() {
    if common::qdp_engine_probed().is_none() || !qdp_core::cuda_runtime_available() {
        println!("SKIP: no engine or CUDA runtime unavailable");
        return;
    }
    let Some(engine_f64) = common::qdp_engine_with_precision(Precision::Float64) else {
        println!("SKIP: no f64 engine available");
        return;
    };

    let message = PipelineIterator::new_synthetic(engine_f64, oversized_config())
        .err()
        .expect("oversized config must be rejected")
        .to_string();

    assert!(
        message.contains("dtype=Float32") && message.contains("device_precision=Float64"),
        "message must show the f32 request budgeted as f64, got: {message}"
    );
    // Twice the f32 figure asserted above: complex128 is 16 bytes per element.
    let single_buffer_mib = 64.0 * (1u64 << 30) as f64 * 16.0 / (1024.0 * 1024.0);
    let expected = format!("requested {:.2} MiB", 2.0 * single_buffer_mib);
    assert!(
        message.contains(&expected),
        "expected the f64 budget ({expected}), got: {message}"
    );
}

/// A configuration that fits must still construct. On a stub CUDA runtime this is the
/// graceful-degradation case: the engine exists via the driver API, the guard short-circuits, and
/// construction proceeds instead of failing on a memory query that cannot succeed.
#[test]
#[cfg(target_os = "linux")]
fn modest_config_constructs_on_any_build() {
    let Some(engine) = common::qdp_engine_probed() else {
        println!("SKIP: No GPU available");
        return;
    };

    let config = PipelineConfig {
        num_qubits: 10,
        batch_size: 8,
        total_batches: 1,
        encoding: Encoding::Amplitude,
        dtype: Precision::Float32,
        prefetch_depth: 1,
        ..PipelineConfig::default()
    };

    assert!(
        PipelineIterator::new_synthetic(engine, config).is_ok(),
        "a 10-qubit batch-8 pipeline must not be rejected by the memory guard"
    );
}
