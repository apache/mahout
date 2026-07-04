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

use std::env;
use std::process::Command;

fn main() {
    emit_gpu_platform_cfg();
    compile_protos();
    configure_cuda_linkage();
}

/// Emit `qdp_gpu_platform` on any OS where the GPU stack is compiled.
///
/// Linux always has it (the original target). Windows gets it when the `hip`
/// feature is active (TheRock-based ROCm; the feature is set by QDP_USE_HIP=1).
/// Source that was `#[cfg(target_os = "linux")]` uses `#[cfg(qdp_gpu_platform)]`
/// so it compiles on both.
fn emit_gpu_platform_cfg() {
    println!("cargo::rustc-check-cfg=cfg(qdp_gpu_platform)");
    let is_linux = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let is_windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    let hip_feature = env::var("CARGO_FEATURE_HIP").is_ok();
    if is_linux || (is_windows && hip_feature) {
        println!("cargo::rustc-cfg=qdp_gpu_platform");
    }
}

fn compile_protos() {
    // Use vendored protoc to avoid missing protoc in CI/dev environments
    unsafe {
        env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().unwrap());
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

/// Detect the CUDA Runtime toolkit and emit the appropriate link directives.
///
/// `qdp-core` declares CUDA Runtime API extern symbols in `src/gpu/cuda_ffi.rs`
/// (cudaHostAlloc, cudaMemGetInfo, cudaEventCreateWithFlags, ...). Those symbols
/// must be resolved at link time, which requires `libcudart` from the CUDA
/// Toolkit. Previously the only `-lcudart` directive lived in `qdp-kernels`'
/// build script, and the `qdp_no_cuda` cfg it sets does not propagate
/// cross-crate. The result was confusing linker errors on systems that have
/// the NVIDIA driver but not the toolkit (e.g. PyTorch-only setups, where
/// PyTorch ships its own bundled cudart inside the wheel).
///
/// This function:
///   * emits `cargo:rustc-link-lib=cudart` and the appropriate
///     `cargo:rustc-link-search` path when nvcc is found, and
///   * emits `cargo:rustc-cfg=qdp_no_cuda` when it is not, gating the
///     extern block in `cuda_ffi.rs` so the crate still links (with stubs
///     that return a non-zero CUDA error at runtime).
fn configure_cuda_linkage() {
    println!("cargo::rustc-check-cfg=cfg(qdp_no_cuda)");
    println!("cargo:rerun-if-env-changed=QDP_NO_CUDA");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    // The whole CUDA-vs-stub decision hinges on finding nvcc on PATH, so a PATH
    // change (e.g. installing the toolkit) must re-trigger this script.
    println!("cargo:rerun-if-env-changed=PATH");

    let force_no_cuda = env::var("QDP_NO_CUDA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false);

    let has_cuda = !force_no_cuda
        && Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

    if !has_cuda {
        println!("cargo:rustc-cfg=qdp_no_cuda");
        println!(
            "cargo:warning=qdp-core: CUDA toolkit not found (nvcc not in PATH). \
             Building with stub CUDA Runtime symbols; GPU functionality will be \
             unavailable at runtime. Install the CUDA Toolkit to enable GPU support."
        );
        return;
    }

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");

    // On macOS, also check /usr/local/cuda/lib
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-search=native={}/lib", cuda_path);
}
