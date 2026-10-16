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

// Build script for the CUDA kernels.
//
// Every `src/*.cu` file is compiled by `nvcc` into a device-only fatbin
// (cubins for each supported architecture plus PTX for forward
// compatibility). The fatbins are embedded into the crate and loaded through
// the CUDA driver API at runtime by `qdp_kernels::registry`, so no host-side
// launcher code, `extern "C"` declaration, or link-time CUDA dependency is
// needed for a kernel.
//
// The script also generates:
//   - the list of `extern "C" __global__` symbols per module, so a test can
//     prove every kernel resolves in its fatbin;
//   - Rust constants mirroring `src/kernel_config.h`, so launch geometry is
//     defined once.
//
// Without `nvcc` (macOS, CI without the toolkit, `QDP_NO_CUDA=1`) the module
// table is empty and every kernel lookup reports `KernelError::Unavailable`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const KERNEL_SOURCES: &[&str] = &["amplitude", "basis", "angle", "validation", "iqp", "phase"];

const DEFAULT_CUBIN_ARCHES: &[&str] = &["75", "80", "86", "89", "90", "100", "120"];
const DEFAULT_PTX_CANDIDATES: &[&str] = &["120", "100", "90", "89", "86", "80", "75"];
const LEGACY_FALLBACK_ARCHES: &[&str] = &["75", "80", "86"];

fn sm_target(arch: &str) -> [String; 2] {
    [
        "-gencode".into(),
        format!("arch=compute_{arch},code=sm_{arch}"),
    ]
}

fn ptx_target(arch: &str) -> [String; 2] {
    [
        "-gencode".into(),
        format!("arch=compute_{arch},code=compute_{arch}"),
    ]
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

fn env_arch_flags(raw: &str) -> Result<Vec<String>, String> {
    let mut flags = Vec::new();
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
            flags.extend(sm_target(&arch));
            flags.extend(ptx_target(&arch));
            continue;
        }

        let arch = parse_arch_name(trimmed)?;
        flags.extend(sm_target(&arch));
    }

    if flags.is_empty() {
        return Err(
            "QDP_CUDA_ARCH_LIST did not contain any usable CUDA architectures after parsing."
                .to_string(),
        );
    }

    Ok(flags)
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

fn default_arch_flags() -> Vec<String> {
    let supported_sm = query_nvcc_list("--list-gpu-code");
    let supported_compute = query_nvcc_list("--list-gpu-arch");
    let mut flags = Vec::new();

    if supported_sm.is_empty() && supported_compute.is_empty() {
        for arch in LEGACY_FALLBACK_ARCHES {
            flags.extend(sm_target(arch));
        }
        return flags;
    }

    let cubin_arches = if supported_sm.is_empty() {
        &supported_compute
    } else {
        &supported_sm
    };
    let mut added_cubin = false;

    for arch in DEFAULT_CUBIN_ARCHES {
        if nvcc_supports(cubin_arches, arch) {
            flags.extend(sm_target(arch));
            added_cubin = true;
        }
    }

    if !added_cubin {
        for arch in LEGACY_FALLBACK_ARCHES {
            flags.extend(sm_target(arch));
        }
    }

    if let Some(ptx_arch) = DEFAULT_PTX_CANDIDATES
        .iter()
        .find(|arch| nvcc_supports(&supported_compute, arch))
    {
        flags.extend(ptx_target(ptx_arch));
    }

    flags
}

fn arch_flags() -> Vec<String> {
    match env::var("QDP_CUDA_ARCH_LIST") {
        Ok(raw) => env_arch_flags(&raw).unwrap_or_else(|message| panic!("{message}")),
        Err(_) => default_arch_flags(),
    }
}

/// Every kernel symbol in a source file, in file order.
///
/// A kernel must be declared as `extern "C" __global__ void NAME(` on one
/// line, optionally with `__launch_bounds__(...)` before the name, so its
/// symbol is unmangled and this parser can see it. Any other `__global__`
/// declaration fails the build with an explanation rather than producing a
/// kernel the registry cannot find at runtime.
fn kernel_symbols(path: &str, source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for (lineno, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || !trimmed.contains("__global__") {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("extern \"C\" __global__ void ") else {
            panic!(
                "{path}:{}: kernel must be declared as `extern \"C\" __global__ void NAME(` on one \
                 line so the registry can resolve it by name; got: {}",
                lineno + 1,
                trimmed
            );
        };
        let rest = rest.trim_start();
        let rest = match rest.strip_prefix("__launch_bounds__") {
            Some(after) => {
                let close = after.find(')').unwrap_or_else(|| {
                    panic!("{path}:{}: unterminated __launch_bounds__", lineno + 1)
                });
                after[close + 1..].trim_start()
            }
            None => rest,
        };
        let name_end = rest.find('(').unwrap_or_else(|| {
            panic!(
                "{path}:{}: kernel name and `(` must be on the declaration line",
                lineno + 1
            )
        });
        let name = rest[..name_end].trim();
        assert!(
            !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "{path}:{}: could not parse kernel name from: {trimmed}",
            lineno + 1
        );
        names.push(name.to_string());
    }
    names
}

/// `#define NAME 123` lines from `kernel_config.h` as `(name, value)`.
fn config_constants(header: &str) -> Vec<(String, u64)> {
    header
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("#define ")?;
            let mut parts = rest.split_whitespace();
            let name = parts.next()?;
            let value = parts.next()?.parse::<u64>().ok()?;
            Some((name.to_string(), value))
        })
        .collect()
}

/// The host compiler nvcc should drive, when the environment names one.
///
/// CI installs a CUDA-compatible gcc and points `CXX`/`CC` at it; nvcc
/// would otherwise pick the system default and refuse newer versions.
fn host_compiler() -> Option<String> {
    ["NVCC_CCBIN", "CXX", "CC"]
        .iter()
        .find_map(|var| env::var(var).ok().filter(|v| !v.trim().is_empty()))
}

fn compile_fatbin(cuda_path: &str, flags: &[String], src: &Path, out: &Path) {
    let mut cmd = Command::new("nvcc");
    if let Some(ccbin) = host_compiler() {
        cmd.arg("-ccbin").arg(ccbin);
    }
    let status = cmd
        .arg("-fatbin")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-I")
        .arg("src")
        .arg("-I")
        .arg(format!("{cuda_path}/include"))
        .args(flags)
        .arg("-o")
        .arg(out)
        .arg(src)
        .status()
        .unwrap_or_else(|e| panic!("failed to run nvcc for {}: {e}", src.display()));
    assert!(
        status.success(),
        "nvcc failed to compile {} to a fatbin",
        src.display()
    );
}

/// Transitional: the hand-written `extern "C"` host launchers still linked by
/// `qdp-core` are compiled and statically linked here. This goes away once every
/// encoder launches through `qdp_kernels::registry`.
fn compile_static_launchers(cuda_path: &str, flags: &[String]) {
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-search=native={}/lib", cuda_path);

    let mut build = cc::Build::new();
    build.include(format!("{}/include", cuda_path));
    build.include("src");
    build.cuda(true).flag("-cudart=shared").flag("-std=c++17");
    for flag in flags {
        build.flag(flag);
    }
    for name in KERNEL_SOURCES {
        build.file(format!("src/{name}.cu"));
    }
    build.compile("kernels");
}

fn write_generated(out_dir: &Path, has_cuda: bool, fatbins: &[(String, PathBuf, Vec<String>)]) {
    let header = fs::read_to_string("src/kernel_config.h").expect("read kernel_config.h");
    let mut generated = String::new();
    generated.push_str("// @generated by build.rs -- do not edit.\n\n");

    generated.push_str("/// Constants mirrored from `src/kernel_config.h`.\npub mod config {\n");
    for (name, value) in config_constants(&header) {
        generated.push_str(&format!("    pub const {name}: usize = {value};\n"));
    }
    generated.push_str("}\n\n");

    generated.push_str(
        "/// Embedded kernel modules: `(module name, fatbin image, kernel symbols)`.\n\
         pub const MODULES: &[(&str, &[u8], &[&str])] = &[\n",
    );
    if has_cuda {
        for (name, path, symbols) in fatbins {
            let symbol_list = symbols
                .iter()
                .map(|s| format!("\"{s}\""))
                .collect::<Vec<_>>()
                .join(", ");
            generated.push_str(&format!(
                "    (\"{name}\", include_bytes!({path:?}), &[{symbol_list}]),\n"
            ));
        }
    }
    generated.push_str("];\n");

    fs::write(out_dir.join("embedded.rs"), generated).expect("write embedded.rs");
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(qdp_no_cuda)");
    for name in KERNEL_SOURCES {
        println!("cargo:rerun-if-changed=src/{name}.cu");
    }
    println!("cargo:rerun-if-changed=src/kernel_config.h");
    println!("cargo:rerun-if-env-changed=QDP_NO_CUDA");
    println!("cargo:rerun-if-env-changed=QDP_CUDA_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=CC");
    // The whole CUDA-vs-stub decision hinges on finding nvcc on PATH, so a PATH
    // change (e.g. installing the toolkit) must re-trigger this script.
    println!("cargo:rerun-if-env-changed=PATH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

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
        println!("cargo:warning=CUDA not found (nvcc not in PATH). Skipping kernel compilation.");
        println!("cargo:warning=This is expected on macOS or non-CUDA environments.");
        println!(
            "cargo:warning=The project will build, but GPU functionality will not be available."
        );
        write_generated(&out_dir, false, &[]);
        return;
    }

    // Priority: CUDA_PATH env var > /usr/local/cuda (default Linux location)
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let flags = arch_flags();
    compile_static_launchers(&cuda_path, &flags);

    let mut fatbins = Vec::new();
    for name in KERNEL_SOURCES {
        let src = PathBuf::from(format!("src/{name}.cu"));
        let out = out_dir.join(format!("{name}.fatbin"));
        compile_fatbin(&cuda_path, &flags, &src, &out);
        let source = fs::read_to_string(&src).expect("read kernel source");
        fatbins.push((
            name.to_string(),
            out,
            kernel_symbols(&src.display().to_string(), &source),
        ));
    }
    write_generated(&out_dir, true, &fatbins);
}
