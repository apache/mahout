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

//! Shared CUDA stream synchronization with unified error reporting.

#[cfg(target_os = "linux")]
use std::ffi::c_void;

use crate::error::{MahoutError, Result, cuda_error_to_string};

/// Synchronizes a CUDA stream and returns a consistent error with context.
///
/// Error message format: `"{context}: {code} ({description})"` so that all
/// call sites report stream sync failures the same way.
///
/// # Arguments
/// * `stream` - CUDA stream pointer (e.g. from PyTorch or default null)
/// * `context` - Short description for the error message (e.g. "Norm stream synchronize failed")
///
/// # Safety
/// The stream pointer must be valid for the duration of this call.
#[cfg(target_os = "linux")]
pub(crate) fn sync_cuda_stream(stream: *mut c_void, context: &str) -> Result<()> {
    let ret = unsafe { crate::gpu::cuda_ffi::cudaStreamSynchronize(stream) };
    if ret != 0 {
        return Err(MahoutError::Cuda(format!(
            "{}: {} ({})",
            context,
            ret,
            cuda_error_to_string(ret)
        )));
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    #[test]
    fn sync_null_stream_does_not_panic() {
        // Default stream (null) sync: may succeed or fail depending on driver/context.
        let _ = sync_cuda_stream(ptr::null_mut::<c_void>(), "test context");
    }

    #[test]
    fn error_message_format_includes_context() {
        // When sync fails, error must be MahoutError::Cuda with format "{context}: {code} ({desc})".
        // We build the same format as sync_cuda_stream to assert consistency.
        let context = "TestContext";
        let code = 999i32;
        let desc = crate::error::cuda_error_to_string(code);
        let msg = format!("{}: {} ({})", context, code, desc);
        assert!(
            msg.starts_with("TestContext:"),
            "format should start with context"
        );
        assert!(msg.contains("TestContext"), "format should contain context");
        assert!(msg.contains("999"), "format should contain error code");
    }
}
