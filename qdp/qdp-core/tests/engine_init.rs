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

//! `QdpEngine::new` must report a missing GPU as an error, never panic.
//!
//! These tests run on every machine: on a GPU host they check the engine
//! comes up; on a host without an NVIDIA driver they check the constructor
//! returns `Err` (it used to panic inside cudarc while loading libcuda,
//! which aborts the Python process in the `_qdp` extension).

use qdp_core::{MahoutError, Precision, QdpEngine, cuda_runtime_available};

#[test]
fn new_agrees_with_runtime_probe() {
    let available = cuda_runtime_available();
    match QdpEngine::new(0) {
        Ok(_) => assert!(available, "engine created but probe says no CUDA"),
        Err(e) => {
            assert!(!available, "probe says CUDA is usable, but: {e}");
            assert!(matches!(e, MahoutError::Cuda(_)), "unexpected error: {e:?}");
            assert!(e.to_string().contains("no usable CUDA device"), "{e}");
        }
    }
}

#[test]
fn new_with_precision_never_panics() {
    for precision in [Precision::Float32, Precision::Float64] {
        let _ = QdpEngine::new_with_precision(0, precision);
    }
}

#[test]
fn out_of_range_device_is_an_error() {
    assert!(QdpEngine::new(usize::MAX >> 1).is_err());
}
