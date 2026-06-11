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
//
// Portions Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>

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

const KERNEL_SOURCES: &[&str] = &[
    "src/amplitude.cu",
    "src/basis.cu",
    "src/angle.cu",
    "src/validation.cu",
    "src/iqp.cu",
    "src/phase.cu",
];

/// Default AMD GPU target used only when QDP_HIP_ARCH_LIST is unset. Never hardcode
/// this as the sole arch in a way that overrides the env list: other AMD targets
/// (gfx1100, gfx1151) must build the same source by setting QDP_HIP_ARCH_LIST alone.
const DEFAULT_HIP_ARCH: &str = "gfx90a";

const DEFAULT_CUBIN_ARCHES: &[&str] = &["75", "80", "86", "89", "90", "100", "120"];
const DEFAULT_PTX_CANDIDATES: &[&str] = &["120", "100", "90", "89", "86", "80", "75"];
const LEGACY_FALLBACK_ARCHES: &[&str] = &["75", "80", "86"];

fn add_sm_target(build: &mut cc::Build, arch: &str) {
    build.flag("-gencode");
    build.flag(format!("arch=compute_{arch},code=sm_{arch}"));
}

fn add_ptx_target(build: &mut cc::Build, arch: &str) {
    build.flag("-gencode");
    build.flag(format!("arch=compute_{arch},code=compute_{arch}"));
}

fn parse_arch_name(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || !trimmed
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        || !trimmed.chars().next().is_some_and(|ch| ch.is_ascii_digit())
    {
        return Err(format!(
            "Invalid CUDA architecture '{trimmed}' in QDP_CUDA_ARCH_LIST. Expected entries like \
             '89', '90a', or '120+PTX'."
        ));
    }

    Ok(trimmed.to_ascii_lowercase())
}

fn apply_env_arch_list(build: &mut cc::Build, raw: &str) -> Result<(), String> {
    let mut saw_target = false;
    for entry in raw.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(base) = trimmed
            .strip_suffix("+PTX")
            .or_else(|| trimmed.strip_suffix("+ptx"))
        {
            let arch = parse_arch_name(base)?;
            add_sm_target(build, &arch);
            add_ptx_target(build, &arch);
            saw_target = true;
            continue;
        }

        let arch = parse_arch_name(trimmed)?;
        add_sm_target(build, &arch);
        saw_target = true;
    }

    if !saw_target {
        return Err(
            "QDP_CUDA_ARCH_LIST did not contain any usable CUDA architectures after parsing."
                .to_string(),
        );
    }

    Ok(())
}

fn query_nvcc_list(flag: &str) -> Vec<String> {
    let Ok(output) = Command::new("nvcc").arg(flag).output() else {
        return Vec::new();
    };

    if !output.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("sm_")
                .or_else(|| line.trim().strip_prefix("compute_"))
        })
        .map(|suffix| suffix.to_ascii_lowercase())
        .collect()
}

fn nvcc_supports(supported_arches: &[String], arch: &str) -> bool {
    supported_arches.iter().any(|supported| supported == arch)
}

fn apply_default_arch_targets(build: &mut cc::Build) {
    let supported_sm = query_nvcc_list("--list-gpu-code");
    let supported_compute = query_nvcc_list("--list-gpu-arch");

    if supported_sm.is_empty() && supported_compute.is_empty() {
        for arch in LEGACY_FALLBACK_ARCHES {
            add_sm_target(build, arch);
        }
        return;
    }

    let cubin_arches = if supported_sm.is_empty() {
        &supported_compute
    } else {
        &supported_sm
    };
    let mut added_cubin = false;

    for arch in DEFAULT_CUBIN_ARCHES {
        if nvcc_supports(cubin_arches, arch) {
            add_sm_target(build, arch);
            added_cubin = true;
        }
    }

    if !added_cubin {
        for arch in LEGACY_FALLBACK_ARCHES {
            add_sm_target(build, arch);
        }
    }

    if let Some(ptx_arch) = DEFAULT_PTX_CANDIDATES
        .iter()
        .find(|arch| nvcc_supports(&supported_compute, arch))
    {
        add_ptx_target(build, ptx_arch);
    }
}

fn hip_requested() -> bool {
    if cfg!(feature = "hip") {
        return true;
    }
    env::var("QDP_USE_HIP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false)
}

/// Compile the kernels with hipcc for AMD GPUs.
///
/// Mirrors the CUDA branch in spirit (same six .cu sources, same `src/` include
/// for kernel_config.h) but: targets are AMD `--offload-arch` values from
/// QDP_HIP_ARCH_LIST (default gfx90a only when unset), the hip_compat/ shim dir
/// is added to the include path so the sources' `<cuda_runtime.h>` /
/// `<cuComplex.h>` / `<vector_types.h>` resolve to HIP equivalents, and the
/// link library is amdhip64 instead of cudart. The CUDA path is untouched.
fn build_hip() {
    let hipcc = env::var("QDP_HIPCC").unwrap_or_else(|_| "hipcc".to_string());

    let mut build = cc::Build::new();
    build.compiler(&hipcc);
    build.cpp(true);
    // hip_compat/ first so its cuda_runtime.h / cuComplex.h / vector_types.h win;
    // src/ for kernel_config.h and kernel_compat.h.
    build.include("hip_compat");
    build.include("src");
    build.flag("-std=c++17");
    build.flag("-x").flag("hip");

    let arch_list =
        env::var("QDP_HIP_ARCH_LIST").unwrap_or_else(|_| DEFAULT_HIP_ARCH.to_string());
    let mut saw_arch = false;
    for entry in arch_list.split(',') {
        let arch = entry.trim();
        if arch.is_empty() {
            continue;
        }
        build.flag(format!("--offload-arch={arch}"));
        saw_arch = true;
    }
    if !saw_arch {
        build.flag(format!("--offload-arch={DEFAULT_HIP_ARCH}"));
    }

    for src in KERNEL_SOURCES {
        build.file(src);
    }
    build.compile("kernels");

    // Link the HIP runtime. Honor an explicit ROCM_PATH; otherwise rely on the
    // default loader search path (hipcc-built objects pull libamdhip64 there).
    if let Ok(rocm) = env::var("ROCM_PATH") {
        println!("cargo:rustc-link-search=native={rocm}/lib");
    }
    println!("cargo:rustc-link-lib=amdhip64");
}

fn main() {
    // Let rustc know about our build-script-defined cfg flags (avoids `unexpected_cfgs` warnings).
    println!("cargo::rustc-check-cfg=cfg(qdp_no_cuda)");
    // Emit qdp_gpu_platform when building for a GPU-capable OS (Linux always;
    // Windows when the hip feature is on via QDP_USE_HIP=1 / TheRock ROCm).
    println!("cargo::rustc-check-cfg=cfg(qdp_gpu_platform)");
    let is_linux = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let hip_feature = std::env::var("CARGO_FEATURE_HIP").is_ok();
    let is_windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if is_linux || (is_windows && hip_feature) {
        println!("cargo::rustc-cfg=qdp_gpu_platform");
    }

    // Tell Cargo to rerun this script if the kernel sources change
    println!("cargo:rerun-if-changed=src/amplitude.cu");
    println!("cargo:rerun-if-changed=src/basis.cu");
    println!("cargo:rerun-if-changed=src/angle.cu");
    println!("cargo:rerun-if-changed=src/validation.cu");
    println!("cargo:rerun-if-changed=src/iqp.cu");
    println!("cargo:rerun-if-changed=src/phase.cu");
    println!("cargo:rerun-if-env-changed=QDP_NO_CUDA");
    println!("cargo:rerun-if-env-changed=QDP_CUDA_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=QDP_USE_HIP");
    println!("cargo:rerun-if-env-changed=QDP_HIP_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=QDP_HIPCC");
    println!("cargo:rerun-if-changed=src/kernel_config.h");
    println!("cargo:rerun-if-changed=src/kernel_compat.h");
    println!("cargo:rerun-if-changed=hip_compat/cuda_runtime.h");
    println!("cargo:rerun-if-changed=hip_compat/cuComplex.h");
    println!("cargo:rerun-if-changed=hip_compat/vector_types.h");

    // AMD/HIP build path: compile the same .cu sources with hipcc. Gated by the
    // `hip` Cargo feature or QDP_USE_HIP=1; the CUDA path below is unchanged when off.
    if hip_requested() {
        build_hip();
        return;
    }

    // Check if CUDA is available by looking for nvcc
    let force_no_cuda = env::var("QDP_NO_CUDA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false);

    let has_cuda = !force_no_cuda && Command::new("nvcc").arg("--version").output().is_ok();

    if !has_cuda {
        // Expose a cfg for conditional compilation of stub symbols on Linux.
        // This allows qdp-kernels (and dependents) to link on Linux machines without CUDA.
        println!("cargo:rustc-cfg=qdp_no_cuda");
        println!("cargo:warning=CUDA not found (nvcc not in PATH). Skipping kernel compilation.");
        println!("cargo:warning=This is expected on macOS or non-CUDA environments.");
        println!(
            "cargo:warning=The project will build, but GPU functionality will not be available."
        );
        println!("cargo:warning=For production deployment, ensure CUDA toolkit is installed.");
        return;
    }

    // Get CUDA installation path
    // Priority: CUDA_PATH env var > /usr/local/cuda (default Linux location)
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");

    // On macOS, also check /usr/local/cuda/lib
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-search=native={}/lib", cuda_path);

    // Compile CUDA kernels
    // This uses cc crate's CUDA support to invoke nvcc
    let mut build = cc::Build::new();

    build.include(format!("{}/include", cuda_path));
    build.include("src"); // Include src directory for kernel_config.h

    build
        .cuda(true)
        .flag("-cudart=shared") // Use shared CUDA runtime
        .flag("-std=c++17"); // C++17 for modern CUDA features

    if let Ok(raw) = env::var("QDP_CUDA_ARCH_LIST") {
        apply_env_arch_list(&mut build, &raw).unwrap_or_else(|message| panic!("{message}"));
    } else {
        apply_default_arch_targets(&mut build);
    }

    build
        .file("src/amplitude.cu")
        .file("src/basis.cu")
        .file("src/angle.cu")
        .file("src/validation.cu")
        .file("src/iqp.cu")
        .file("src/phase.cu")
        .compile("kernels");
}
