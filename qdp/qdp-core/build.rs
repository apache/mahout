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
    // Emit qdp_gpu_platform cfg on any OS where the GPU stack is compiled.
    // Linux always has it (original target). Windows gets it when the `hip`
    // feature is active (TheRock-based ROCm; the feature is set by QDP_USE_HIP=1).
    // Source code that was `#[cfg(target_os = "linux")]` should use
    // `#[cfg(qdp_gpu_platform)]` so it compiles on both.
    println!("cargo::rustc-check-cfg=cfg(qdp_gpu_platform)");
    let is_linux = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let hip_feature = std::env::var("CARGO_FEATURE_HIP").is_ok();
    let is_windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if is_linux || (is_windows && hip_feature) {
        println!("cargo::rustc-cfg=qdp_gpu_platform");
    }

    // Use vendored protoc to avoid missing protoc in CI/dev environments
    unsafe {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().unwrap());
    }

    let mut config = prost_build::Config::new();

    // Generate tensor_content as bytes::Bytes (avoids copy during protobuf decode)
    config.bytes([".tensorflow.TensorProto.tensor_content"]);

    // Generate fixed filename include file to avoid guessing output filename/module path
    config.include_file("tensorflow_proto_mod.rs");

    config
        .compile_protos(&["proto/tensor.proto"], &["proto"])
        .unwrap();

    println!("cargo:rerun-if-changed=proto/tensor.proto");
}
