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

#[cfg(target_os = "linux")]
pub mod buffer_pool;
#[cfg(target_os = "linux")]
pub(crate) mod cuda_sync;
pub mod encodings;
pub mod memory;
/// Test-only fidelity / trace-distance helpers. Public so integration tests in
/// `tests/` can use them; not part of the supported runtime API.
#[doc(hidden)]
pub mod metrics;
#[cfg(target_os = "linux")]
pub mod overlap_tracker;
pub mod pipeline;
#[cfg(target_os = "linux")]
pub mod pool_metrics;
#[cfg(target_os = "linux")]
pub(crate) mod validation;

#[cfg(target_os = "linux")]
pub(crate) mod cuda_ffi;

#[cfg(target_os = "linux")]
pub use buffer_pool::{PinnedBufferHandle, PinnedBufferPool};
pub use encodings::{AmplitudeEncoder, AngleEncoder, BasisEncoder, QuantumEncoder};
pub use memory::GpuStateVector;
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub use metrics::{download_complex_f32, download_complex_f64};
#[doc(hidden)]
pub use metrics::{
    fidelity_cross_precision, fidelity_f32, fidelity_f64, trace_distance_cross_precision,
    trace_distance_f32, trace_distance_f64,
};
pub use pipeline::run_dual_stream_pipeline;

#[cfg(target_os = "linux")]
pub use overlap_tracker::OverlapTracker;
#[cfg(target_os = "linux")]
pub use pipeline::PipelineContext;
#[cfg(target_os = "linux")]
pub use pool_metrics::{PoolMetrics, PoolUtilizationReport};
