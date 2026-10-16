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

//! Runtime kernel registry.
//!
//! Kernels are compiled to device-only fatbins by `build.rs`, embedded in this
//! crate, and loaded through the CUDA driver API the first time a kernel is
//! requested on a device. After that a lookup is a hash probe under a read
//! lock; the driver call is the same `cuLaunchKernel` that `<<<>>>` expands
//! to, so there is no per-launch cost compared with a statically linked
//! launcher.
//!
//! A kernel author writes device code only. The Rust side names the kernel,
//! chooses the launch geometry, and passes arguments with [`kernel_args!`]:
//!
//! ```ignore
//! let f = registry::function(&device, "angle", "angle_encode_kernel_f32")?;
//! unsafe {
//!     registry::launch(
//!         &device,
//!         f,
//!         LaunchConfig::grid_1d(state_len),
//!         stream,
//!         &mut kernel_args![angles_ptr, state_ptr, state_len, num_qubits],
//!     )?;
//! }
//! ```
//!
//! Argument types must match the kernel signature exactly: `*const T` /
//! `*mut T` for device pointers, `usize` for `size_t`, `u32` for
//! `unsigned int`, `i32` for `int`, `f64` for `double`, `f32` for `float`.

use std::ffi::c_void;
use std::fmt;

#[cfg(target_os = "linux")]
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::ffi::CString;
#[cfg(target_os = "linux")]
use std::sync::{OnceLock, RwLock};

use cudarc::driver::{CudaDevice, DriverError};
#[cfg(target_os = "linux")]
use cudarc::driver::{result, sys};

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

/// Why a kernel could not be resolved or launched.
#[derive(Debug)]
pub enum KernelError {
    /// The crate was built without the CUDA toolkit, so no kernels are embedded.
    Unavailable,
    /// No embedded module has this name.
    UnknownModule(String),
    /// The module exists but has no kernel with this symbol.
    UnknownSymbol { module: String, name: String },
    /// The CUDA driver rejected the load or launch.
    Driver(DriverError),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(
                f,
                "CUDA kernels unavailable (qdp-kernels was built without the CUDA toolkit)"
            ),
            Self::UnknownModule(m) => write!(f, "no embedded kernel module named '{m}'"),
            Self::UnknownSymbol { module, name } => {
                write!(f, "kernel module '{module}' has no symbol '{name}'")
            }
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
        }
    }
}

impl std::error::Error for KernelError {}

impl From<DriverError> for KernelError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// A resolved kernel handle. Cheap to copy. It is bound to the primary
/// context that was current when it was resolved; if that context has since
/// been destroyed (every `CudaDevice` for the GPU dropped), a launch through
/// it reloads the module transparently.
#[derive(Clone, Copy, Debug)]
pub struct Function {
    #[cfg(target_os = "linux")]
    raw: sys::CUfunction,
    #[cfg(target_os = "linux")]
    ctx: sys::CUcontext,
    module: &'static str,
    name: &'static str,
}

// A CUfunction is an opaque handle owned by the driver; it carries no
// thread affinity, and every launch re-binds the device context.
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

/// One-dimensional launch geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: u32,
    pub block: u32,
    pub shared_bytes: u32,
}

impl LaunchConfig {
    /// One thread per element, `DEFAULT_BLOCK_SIZE` threads per block, as
    /// many blocks as needed (at least one).
    pub fn grid_1d(elements: usize) -> Self {
        let block = config::DEFAULT_BLOCK_SIZE;
        let grid = elements.div_ceil(block).max(1);
        Self {
            grid: grid as u32,
            block: block as u32,
            shared_bytes: 0,
        }
    }

    /// Grid-stride geometry: blocks capped at `max_blocks` so a kernel that
    /// loops over its input keeps the GPU full without oversubscribing.
    pub fn grid_stride(elements: usize, max_blocks: usize) -> Self {
        let block = config::DEFAULT_BLOCK_SIZE;
        let grid = elements.div_ceil(block).clamp(1, max_blocks);
        Self {
            grid: grid as u32,
            block: block as u32,
            shared_bytes: 0,
        }
    }

    pub fn exact(grid: u32, block: u32) -> Self {
        Self {
            grid,
            block,
            shared_bytes: 0,
        }
    }

    pub fn with_shared_bytes(mut self, bytes: usize) -> Self {
        self.shared_bytes = bytes as u32;
        self
    }
}

/// Build the argument array for [`launch`]. Construct it inline in the
/// `launch` call so the referenced values outlive the launch:
/// `launch(.., &mut kernel_args![a, b, c])`.
#[macro_export]
macro_rules! kernel_args {
    ($($arg:expr),* $(,)?) => {
        [ $( &$arg as *const _ as *mut ::std::ffi::c_void ),* ]
    };
}

#[cfg(target_os = "linux")]
struct ModuleHandle(sys::CUmodule);
#[cfg(target_os = "linux")]
unsafe impl Send for ModuleHandle {}
#[cfg(target_os = "linux")]
unsafe impl Sync for ModuleHandle {}

/// Caches are keyed by the primary context the handles belong to, not just
/// the ordinal: cudarc destroys a GPU's primary context when the last
/// `CudaDevice` for it is dropped, which invalidates every module and
/// function handle loaded into it.
#[cfg(target_os = "linux")]
type CtxKey = (usize, usize);
#[cfg(target_os = "linux")]
type ModuleKey = (CtxKey, &'static str);
#[cfg(target_os = "linux")]
type FunctionKey = (CtxKey, &'static str, &'static str);

#[cfg(target_os = "linux")]
static MODULES_LOADED: OnceLock<RwLock<HashMap<ModuleKey, ModuleHandle>>> = OnceLock::new();
#[cfg(target_os = "linux")]
static FUNCTIONS: OnceLock<RwLock<HashMap<FunctionKey, Function>>> = OnceLock::new();

#[cfg(target_os = "linux")]
fn ctx_key(device: &CudaDevice) -> CtxKey {
    (device.ordinal(), *device.cu_primary_ctx() as usize)
}

/// Names of the embedded modules.
pub fn module_names() -> impl Iterator<Item = &'static str> {
    MODULES.iter().map(|(name, _, _)| *name)
}

/// Kernel symbols declared in an embedded module, in source order.
pub fn module_symbols(module: &str) -> Option<&'static [&'static str]> {
    MODULES
        .iter()
        .find(|(name, _, _)| *name == module)
        .map(|(_, _, symbols)| *symbols)
}

/// True when this build embeds kernels (the CUDA toolkit was present).
pub fn kernels_embedded() -> bool {
    !MODULES.is_empty()
}

/// Resolve `name` in embedded module `module` on `device`, loading the module
/// on first use.
#[cfg(target_os = "linux")]
pub fn function(
    device: &CudaDevice,
    module: &'static str,
    name: &'static str,
) -> Result<Function, KernelError> {
    let key = ctx_key(device);
    let functions = FUNCTIONS.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(f) = functions
        .read()
        .expect("kernel function cache poisoned")
        .get(&(key, module, name))
    {
        return Ok(*f);
    }
    resolve(device, module, name)
}

/// Load (or re-load) the module and look the symbol up, bypassing the cache.
#[cfg(target_os = "linux")]
fn resolve(
    device: &CudaDevice,
    module: &'static str,
    name: &'static str,
) -> Result<Function, KernelError> {
    let key = ctx_key(device);
    let (_, image, _) = MODULES
        .iter()
        .find(|(n, _, _)| *n == module)
        .ok_or_else(|| {
            if MODULES.is_empty() {
                KernelError::Unavailable
            } else {
                KernelError::UnknownModule(module.to_string())
            }
        })?;

    device.bind_to_thread()?;
    let modules = MODULES_LOADED.get_or_init(|| RwLock::new(HashMap::new()));
    let mut modules = modules.write().expect("kernel module cache poisoned");
    let cname = CString::new(name).expect("kernel symbol contains NUL");
    // A cached module handle may belong to a primary context that was
    // destroyed and re-created at the same address; the driver reports that
    // as INVALID_HANDLE, and the cure is to load the module again.
    let mut raw = None;
    for attempt in 0..2 {
        let handle = match modules.get(&(key, module)) {
            Some(h) => h.0,
            None => {
                // SAFETY: `image` is a complete fatbin produced by nvcc for this
                // crate; the device's primary context is current on this thread.
                let h = unsafe { result::module::load_data(image.as_ptr() as *const c_void)? };
                modules.insert((key, module), ModuleHandle(h));
                h
            }
        };
        // SAFETY: `handle` is a loaded module; the driver reports a missing symbol.
        match unsafe { result::module::get_function(handle, cname.clone()) } {
            Ok(f) => {
                raw = Some(f);
                break;
            }
            Err(DriverError(sys::CUresult::CUDA_ERROR_NOT_FOUND)) => {
                return Err(KernelError::UnknownSymbol {
                    module: module.to_string(),
                    name: name.to_string(),
                });
            }
            Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_HANDLE)) if attempt == 0 => {
                modules.retain(|(k, _), _| *k != key);
                if let Some(f) = FUNCTIONS.get() {
                    f.write()
                        .expect("kernel function cache poisoned")
                        .retain(|(k, _, _), _| *k != key);
                }
            }
            Err(e) => return Err(e.into()),
        }
    }
    let raw = raw.expect("module lookup loop always resolves or returns");
    let f = Function {
        raw,
        ctx: *device.cu_primary_ctx(),
        module,
        name,
    };
    FUNCTIONS
        .get_or_init(|| RwLock::new(HashMap::new()))
        .write()
        .expect("kernel function cache poisoned")
        .insert((key, module, name), f);
    Ok(f)
}

/// Forget every handle cached for `device`'s current primary context.
#[cfg(target_os = "linux")]
fn evict(device: &CudaDevice) {
    let key = ctx_key(device);
    if let Some(m) = MODULES_LOADED.get() {
        m.write()
            .expect("kernel module cache poisoned")
            .retain(|(k, _), _| *k != key);
    }
    if let Some(f) = FUNCTIONS.get() {
        f.write()
            .expect("kernel function cache poisoned")
            .retain(|(k, _, _), _| *k != key);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn function(
    _device: &CudaDevice,
    _module: &'static str,
    _name: &'static str,
) -> Result<Function, KernelError> {
    Err(KernelError::Unavailable)
}

impl Function {
    pub fn module(&self) -> &'static str {
        self.module
    }
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// Launch `f` on `stream` (null for the device's default stream).
///
/// # Safety
/// `args` must match the kernel's parameter list in count, order and ABI
/// type, every device pointer must be valid on `stream`, and the buffers
/// must stay alive until the stream reaches the launch.
#[cfg(target_os = "linux")]
pub unsafe fn launch(
    device: &CudaDevice,
    f: Function,
    cfg: LaunchConfig,
    stream: *mut c_void,
    args: &mut [*mut c_void],
) -> Result<(), KernelError> {
    device.bind_to_thread()?;
    // A handle from a primary context that has since been destroyed and
    // re-created (same pointer, new context) fails with INVALID_HANDLE;
    // reload once and retry.
    let f = if f.ctx == *device.cu_primary_ctx() {
        f
    } else {
        evict(device);
        resolve(device, f.module, f.name)?
    };
    // SAFETY: forwarded from the caller's contract; the context is current.
    let launched = unsafe { launch_raw(f.raw, cfg, stream, args) };
    match launched {
        Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_HANDLE)) => {
            evict(device);
            let f = resolve(device, f.module, f.name)?;
            // SAFETY: as above.
            unsafe { launch_raw(f.raw, cfg, stream, args) }?;
            Ok(())
        }
        other => other.map_err(Into::into),
    }
}

#[cfg(target_os = "linux")]
unsafe fn launch_raw(
    f: sys::CUfunction,
    cfg: LaunchConfig,
    stream: *mut c_void,
    args: &mut [*mut c_void],
) -> Result<(), DriverError> {
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        result::launch_kernel(
            f,
            (cfg.grid, 1, 1),
            (cfg.block, 1, 1),
            cfg.shared_bytes,
            stream as sys::CUstream,
            args,
        )
    }
}

#[cfg(not(target_os = "linux"))]
pub unsafe fn launch(
    _device: &CudaDevice,
    _f: Function,
    _cfg: LaunchConfig,
    _stream: *mut c_void,
    _args: &mut [*mut c_void],
) -> Result<(), KernelError> {
    Err(KernelError::Unavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_1d_rounds_up_and_never_zero() {
        assert_eq!(LaunchConfig::grid_1d(0).grid, 1);
        assert_eq!(LaunchConfig::grid_1d(1).grid, 1);
        let block = config::DEFAULT_BLOCK_SIZE;
        assert_eq!(LaunchConfig::grid_1d(block).grid, 1);
        assert_eq!(LaunchConfig::grid_1d(block + 1).grid, 2);
    }

    #[test]
    fn grid_stride_is_capped() {
        let block = config::DEFAULT_BLOCK_SIZE;
        assert_eq!(LaunchConfig::grid_stride(block * 10_000, 64).grid, 64);
        assert_eq!(LaunchConfig::grid_stride(0, 64).grid, 1);
    }

    #[test]
    fn config_constants_are_generated() {
        assert_eq!(config::DEFAULT_BLOCK_SIZE, 256);
        assert_eq!(config::MAX_GRID_BLOCKS, 2048);
    }
}
