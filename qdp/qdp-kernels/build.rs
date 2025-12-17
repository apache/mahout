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

// Build script for compiling CUDA kernels
//
// This script is executed by Cargo before building the main crate.
// It compiles the .cu files using nvcc and links them with the Rust code.
//
// NOTE: For development environments without CUDA (e.g., macOS), this script
// will detect the absence of nvcc and skip compilation. The project will still
// build, but GPU functionality will not be available.

use std::env;
use std::process::Command;

fn main() {
    // Tell Cargo to rerun this script if the kernel source changes
    println!("cargo:rerun-if-changed=src/amplitude.cu");

    // Check if CUDA is available by looking for nvcc
    let has_cuda = Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok();

    if !has_cuda {
        println!("cargo:warning=CUDA not found (nvcc not in PATH). Skipping kernel compilation.");
        println!("cargo:warning=This is expected on macOS or non-CUDA environments.");
        println!("cargo:warning=The project will build, but GPU functionality will not be available.");
        println!("cargo:warning=For production deployment, ensure CUDA toolkit is installed.");
        return;
    }

    // Get CUDA installation path
    // Priority: CUDA_PATH env var > /usr/local/cuda (default Linux location)
    let cuda_path = env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");

    // On macOS, also check /usr/local/cuda/lib
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-search=native={}/lib", cuda_path);

    // Compile CUDA kernels
    // This uses cc crate's CUDA support to invoke nvcc
    let mut build = cc::Build::new();

    build.include(format!("{}/include", cuda_path));

    build
        .cuda(true)
        .flag("-cudart=shared")  // Use shared CUDA runtime
        .flag("-std=c++17")      // C++17 for modern CUDA features
        // GPU architecture targets
        // SM 75 = Turing (T4, RTX 2000 series)
        // SM 80 = Ampere (A100, RTX 3000 series)
        // SM 86 = Ampere (RTX 3090, A40)
        // SM 89 = Ada Lovelace (RTX 4000 series)
        // SM 90 = Hopper (H100)
        // Support both Turing (sm_75) and Ampere+ architectures
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75")
        .flag("-gencode")
        .flag("arch=compute_80,code=sm_80")
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86")
        // Optional: Add more architectures for production
        // .flag("-gencode")
        // .flag("arch=compute_89,code=sm_89")
        .file("src/amplitude.cu")
        .compile("kernels");
}
