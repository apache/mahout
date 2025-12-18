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
use std::path::{Path, PathBuf};
use std::process::Command;

struct CudaConfig {
    root: PathBuf,
    nvcc: PathBuf,
    include_dir: PathBuf,
    lib_dir: PathBuf,
}

fn main() {
    println!("cargo:rerun-if-changed=src/amplitude.cu");
    println!("cargo:rerun-if-env-changed=QDP_CPU_ONLY");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=PATH");

    if cpu_only_enabled() {
        println!("cargo:rustc-cfg=cpu_only");
        println!("cargo:warning=QDP_CPU_ONLY=1 detected: building in CPU-only mode and skipping CUDA kernels.");
        return;
    }

    let cuda = detect_cuda().unwrap_or_else(|err| {
        eprintln!("{err}");
        eprintln!("Hint: install the CUDA toolkit and ensure nvcc is available, or set QDP_CPU_ONLY=1 to build without GPU kernels.");
        panic!("CUDA toolkit not found");
    });

    println!(
        "cargo:warning=CUDA toolkit detected at {} (nvcc = {}). Building GPU kernels.",
        cuda.root.display(),
        cuda.nvcc.display()
    );

    emit_link_directives(&cuda.lib_dir);

    let mut build = cc::Build::new();
    build.cuda(true)
        .compiler(&cuda.nvcc)
        .include(&cuda.include_dir)
        .flag("-cudart=shared")
        .flag("-std=c++17")
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75")
        .flag("-gencode")
        .flag("arch=compute_80,code=sm_80")
        .file("src/amplitude.cu")
        .compile("kernels");
}

fn cpu_only_enabled() -> bool {
    env::var("QDP_CPU_ONLY")
        .map(|val| val == "1" || val.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn detect_cuda() -> Result<CudaConfig, String> {
    let nvcc = locate_nvcc().ok_or_else(|| {
        format!(
            "nvcc not found. Checked CUDA_HOME, CUDA_PATH, NVCC, and PATH.\
             \nSet CUDA_HOME/CUDA_PATH to your toolkit root (e.g. /usr/local/cuda) so we can find bin/nvcc."
        )
    })?;

    Command::new(&nvcc)
        .arg("--version")
        .output()
        .map_err(|e| format!("Failed to execute nvcc at {}: {:?}", nvcc.display(), e))?;

    let root = nvcc.parent()
        .and_then(Path::parent)
        .ok_or_else(|| format!("Unable to derive CUDA root from nvcc at {}", nvcc.display()))?;

    let include_dir = root.join("include");
    if !include_dir.exists() {
        return Err(format!(
            "CUDA include directory missing at {}. Set CUDA_HOME/CUDA_PATH to a valid toolkit install.",
            include_dir.display()
        ));
    }

    let lib_dir = find_library_dir(root).ok_or_else(|| {
        format!(
            "CUDA library directory not found under {}. Looked for lib64/lib (Linux/macOS) or lib/x64 (Windows).",
            root.display()
        )
    })?;

    Ok(CudaConfig {
        root: root.to_path_buf(),
        nvcc,
        include_dir,
        lib_dir,
    })
}

fn locate_nvcc() -> Option<PathBuf> {
    let nvcc_name = if cfg!(target_os = "windows") { "nvcc.exe" } else { "nvcc" };

    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(nvcc_env) = env::var_os("NVCC") {
        candidates.push(PathBuf::from(nvcc_env));
    }

    if let Some(cuda_home) = env::var_os("CUDA_HOME") {
        candidates.push(PathBuf::from(cuda_home).join("bin").join(nvcc_name));
    }

    if let Some(cuda_path) = env::var_os("CUDA_PATH") {
        candidates.push(PathBuf::from(cuda_path).join("bin").join(nvcc_name));
    }

    if cfg!(not(target_os = "windows")) {
        candidates.push(PathBuf::from("/usr/local/cuda").join("bin").join(nvcc_name));
    }

    candidates.extend(path_nvcc_candidates(nvcc_name));

    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

fn path_nvcc_candidates(nvcc_name: &str) -> Vec<PathBuf> {
    env::var_os("PATH")
        .map(|paths| {
            env::split_paths(&paths)
                .map(|p| p.join(nvcc_name))
                .collect()
        })
        .unwrap_or_default()
}

fn find_library_dir(root: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if cfg!(target_os = "windows") {
        candidates.push(root.join("lib").join("x64"));
        candidates.push(root.join("lib64"));
    } else if cfg!(target_os = "macos") {
        candidates.push(root.join("lib"));
        candidates.push(root.join("lib64"));
    } else {
        candidates.push(root.join("lib64"));
        candidates.push(root.join("lib"));
    }

    candidates.into_iter().find(|p| p.exists())
}

fn emit_link_directives(lib_dir: &Path) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=cudart");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=cudart");
    } else {
        println!("cargo:rustc-link-lib=cudart");
    }
}
