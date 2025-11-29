// Build script for compiling CUDA kernels
//
// This script is executed by Cargo before building the main crate.
// It compiles the .cu files using nvcc and links them with the Rust code.
//
// NOTE: For development environments without CUDA (e.g., macOS), this script
// will detect the absence of nvcc and skip compilation. The project will still
// build, but GPU functionality will not be available.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const ENV_CPU_ONLY: &str = "QDP_CPU_ONLY";
const ENV_ARCHS: &[&str] = &["QDP_CUDA_ARCHS", "CUDA_ARCHS"];
const ENV_CUDA_ROOT: &[&str] = &["CUDA_HOME", "CUDA_PATH"];
const ENV_NVCC: &str = "NVCC";
const ENV_EXTRA_FLAGS: &str = "QDP_NVCC_FLAGS";
const DEFAULT_ARCHS: &[&str] = &["80", "86", "89", "90"]; // Ampere/Lovelace/Hopper coverage

fn main() {
    track_env_changes();

    let kernel_sources = collect_kernel_sources("src");
    if kernel_sources.is_empty() {
        println!("cargo:warning=No CUDA kernel sources found under src/*.cu; skipping compilation.");
        println!("cargo:rustc-cfg=cpu_only");
        return;
    }
    for path in &kernel_sources {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    if env_truthy(ENV_CPU_ONLY) {
        println!("cargo:warning=CPU-only build requested via {}=1; skipping CUDA kernels.", ENV_CPU_ONLY);
        println!("cargo:rustc-cfg=cpu_only");
        return;
    }

    let cuda = match find_nvcc() {
        Some(cfg) => cfg,
        None => {
            println!("cargo:warning=nvcc not found (checked {} envs, default paths, and PATH). Building in CPU-only mode.", ENV_NVCC);
            println!("cargo:warning=Set CUDA_HOME/CUDA_PATH or NVCC to point to your CUDA toolkit (>=12.2).");
            println!("cargo:rustc-cfg=cpu_only");
            return;
        }
    };

    let arch_list = match parse_archs() {
        Ok(list) => list,
        Err(msg) => panic!("Invalid CUDA arch list: {msg}"),
    };

    emit_link_search_paths(&cuda);

    let extra_flags = parse_extra_flags();
    let mut build = cc::Build::new();
    build.cuda(true).flag("-cudart=shared").flag("-std=c++17");

    for arch in &arch_list {
        build.flag("-gencode");
        build.flag(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    for flag in &extra_flags {
        build.flag(flag);
    }
    for path in &kernel_sources {
        build.file(path);
    }

    build.compile("kernels");

    println!(
        "cargo:warning=CUDA kernels compiled (nvcc={}, archs={}, files={})",
        cuda.nvcc.display(),
        arch_list.join("/"),
        kernel_sources.len()
    );
}

struct CudaConfig {
    nvcc: PathBuf,
    cuda_root: PathBuf,
}

fn track_env_changes() {
    println!("cargo:rerun-if-env-changed={ENV_CPU_ONLY}");
    println!("cargo:rerun-if-env-changed={ENV_NVCC}");
    println!("cargo:rerun-if-env-changed={ENV_EXTRA_FLAGS}");
    for key in ENV_ARCHS {
        println!("cargo:rerun-if-env-changed={key}");
    }
    for key in ENV_CUDA_ROOT {
        println!("cargo:rerun-if-env-changed={key}");
    }
}

fn env_truthy(key: &str) -> bool {
    matches!(
        env::var(key)
            .ok()
            .map(|v| v.to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn collect_kernel_sources(dir: &str) -> Vec<PathBuf> {
    let src_dir = Path::new(dir);
    let Ok(entries) = fs::read_dir(src_dir) else {
        println!("cargo:warning=Kernel source directory not found: {}", src_dir.display());
        return Vec::new();
    };

    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "cu").unwrap_or(false))
        .collect();
    files.sort();
    files
}

fn parse_archs() -> Result<Vec<String>, String> {
    for key in ENV_ARCHS {
        if let Ok(val) = env::var(key) {
            let parsed = split_archs(&val)?;
            if parsed.is_empty() {
                return Err(format!("{key} is set but empty"));
            }
            return Ok(parsed);
        }
    }
    Ok(DEFAULT_ARCHS.iter().map(|s| s.to_string()).collect())
}

fn split_archs(raw: &str) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !trimmed.chars().all(|c| c.is_ascii_digit()) {
            return Err(format!("arch '{trimmed}' must be numeric, e.g. 80 or 90"));
        }
        if trimmed.len() < 2 || trimmed.len() > 3 {
            return Err(format!("arch '{trimmed}' should be 2-3 digits (got {})", trimmed.len()));
        }
        result.push(trimmed.to_string());
    }
    Ok(result)
}

fn parse_extra_flags() -> Vec<String> {
    env::var(ENV_EXTRA_FLAGS)
        .ok()
        .map(|v| v.split_whitespace().map(|s| s.to_string()).collect())
        .unwrap_or_default()
}

fn emit_link_search_paths(cfg: &CudaConfig) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());
    match target_os.as_str() {
        "windows" => {
            println!(
                "cargo:rustc-link-search=native={}",
                cfg.cuda_root.join("lib").join("x64").display()
            );
            println!("cargo:rustc-link-lib=cudart");
        }
        "macos" => {
            println!(
                "cargo:rustc-link-search=native={}",
                cfg.cuda_root.join("lib").display()
            );
            println!("cargo:rustc-link-lib=cudart");
        }
        _ => {
            println!(
                "cargo:rustc-link-search=native={}",
                cfg.cuda_root.join("lib64").display()
            );
            println!("cargo:rustc-link-lib=cudart");
        }
    }
}

fn find_nvcc() -> Option<CudaConfig> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(nvcc) = env::var(ENV_NVCC) {
        candidates.push(PathBuf::from(nvcc));
    }

    for key in ENV_CUDA_ROOT {
        if let Ok(root) = env::var(key) {
            candidates.push(Path::new(&root).join("bin").join(nvcc_name()));
        }
    }

    candidates.push(Path::new("/usr/local/cuda").join("bin").join(nvcc_name()));

    if let Ok(path_var) = env::var("PATH") {
        for p in env::split_paths(&path_var) {
            candidates.push(p.join(nvcc_name()));
        }
    }

    for candidate in candidates {
        if nvcc_works(&candidate) {
            let cuda_root = candidate
                .parent()
                .and_then(|p| p.parent())
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
            return Some(CudaConfig {
                nvcc: candidate,
                cuda_root,
            });
        }
    }

    None
}

fn nvcc_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    }
}

fn nvcc_works(path: &Path) -> bool {
    Command::new(path)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
