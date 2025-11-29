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

