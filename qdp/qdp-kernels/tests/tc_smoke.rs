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

//! Smoke tests — verify TC kernel stubs link and return 999 when CUDA unavailable.

#[cfg(any(not(target_os = "linux"), qdp_no_cuda))]
#[test]
fn launch_iqp_encode_tc_stub_returns_unavailable() {
    // SAFETY: stubs ignore all pointer arguments and return 999 immediately.
    #[allow(unused_unsafe)]
    let result = unsafe {
        qdp_kernels::launch_iqp_encode_tc(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 999, "no-CUDA stub must return 999");
}
