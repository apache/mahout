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


// Zero-cost profiling macros for NVTX integration
//
// Provides clean abstraction over NVTX markers without cluttering business logic.
// When observability feature is disabled, these macros compile to no-ops.

/// Profile a scope using RAII guard pattern
///
/// Automatically pushes NVTX range on entry and pops on scope exit.
/// Uses Rust's Drop mechanism to ensure proper cleanup even on early returns.
///
/// # Example
/// ```rust
/// fn my_function() {
///     crate::profile_scope!("MyFunction");
///     // ... code ...
///     // Guard automatically pops when function returns
/// }
/// ```
#[cfg(feature = "observability")]
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _scope_guard = nvtx::range!($name);
    };
}

/// No-op version when observability is disabled
///
/// Compiler eliminates this completely, zero runtime cost.
#[cfg(not(feature = "observability"))]
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        // Zero-cost: compiler removes this entirely
    };
}

/// Mark a point in time with NVTX marker
///
/// Useful for marking specific events without creating a range.
///
/// # Example
/// ```rust
/// crate::profile_mark!("CheckpointReached");
/// ```
#[cfg(feature = "observability")]
#[macro_export]
macro_rules! profile_mark {
    ($name:expr) => {
        nvtx::mark!($name);
    };
}

/// No-op version when observability is disabled
#[cfg(not(feature = "observability"))]
#[macro_export]
macro_rules! profile_mark {
    ($name:expr) => {
        // Zero-cost: compiler removes this entirely
    };
}
