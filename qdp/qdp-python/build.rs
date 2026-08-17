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

fn main() {
    // Emit qdp_gpu_platform when building for a GPU-capable OS (Linux always;
    // Windows when the hip feature is on via QDP_USE_HIP=1 / TheRock ROCm).
    println!("cargo::rustc-check-cfg=cfg(qdp_gpu_platform)");
    let is_linux = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let hip_feature = std::env::var("CARGO_FEATURE_HIP").is_ok();
    let is_windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if is_linux || (is_windows && hip_feature) {
        println!("cargo::rustc-cfg=qdp_gpu_platform");
    }
}
