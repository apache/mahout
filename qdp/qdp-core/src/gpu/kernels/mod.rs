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

//! The one encode path.
//!
//! Every encoding is a [`Kernel`]: a small Rust descriptor that knows its
//! sample width, how to validate input, and how to launch its device code on
//! a batch that is already resident on the GPU. Everything else -- uploading
//! host data through the dual-stream pipeline, allocating the state vector,
//! converting precision, wrapping DLPack -- is shared and lives here or in the
//! engine.
//!
//! A single sample is a batch of one. Host input and device input differ only
//! in who performs the upload. `f32` and `f64` differ only in which kernel
//! symbol is launched. None of that is the kernel author's problem.
//!
//! To add an encoding: write `qdp-kernels/src/<name>.cu` with
//! `extern "C" __global__` kernels that take a batch, implement [`Kernel`] in
//! `kernels/<name>.rs`, and add the variant to [`crate::types::Encoding`].

use std::any::Any;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, DevicePtr as _, DeviceRepr, ValidAsZeroBits};
use qdp_kernels::{CuComplex, CuDoubleComplex, KernelError, LaunchConfig};

use crate::error::{MahoutError, Result};
use crate::gpu::memory::{GpuStateVector, Precision};

pub mod amplitude;
pub mod angle;
pub mod basis;
pub mod iqp;
pub mod phase;

pub use amplitude::AmplitudeKernel;
pub use angle::AngleKernel;
pub use basis::BasisKernel;
pub use iqp::{IqpKernel, iqp_full_kernel, iqp_z_kernel};
pub use phase::PhaseKernel;

/// Practical upper bound on qubits: a 2^30 complex128 state is 16 GB.
pub const MAX_QUBITS: usize = 30;

/// Host inputs at or above this size go through the pinned, double-buffered
/// dual-stream upload instead of one synchronous copy.
pub(crate) const ASYNC_UPLOAD_THRESHOLD_BYTES: usize = 1024 * 1024;

pub fn validate_qubit_count(num_qubits: usize) -> Result<()> {
    if num_qubits == 0 {
        return Err(MahoutError::InvalidInput(
            "Number of qubits must be at least 1".to_string(),
        ));
    }
    if num_qubits > MAX_QUBITS {
        return Err(MahoutError::InvalidInput(format!(
            "Number of qubits {} exceeds practical limit of {}",
            num_qubits, MAX_QUBITS
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------

/// A real element type the kernels are compiled for.
pub trait Real:
    Copy + Send + Sync + Unpin + 'static + DeviceRepr + ValidAsZeroBits + std::fmt::Display + PartialEq
{
    const PRECISION: Precision;
    const ZERO: Self;
    fn is_finite(self) -> bool;
    fn device_ptr(ptr: *const Self) -> DeviceInput;
}

impl Real for f64 {
    const PRECISION: Precision = Precision::Float64;
    const ZERO: Self = 0.0;
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    fn device_ptr(ptr: *const Self) -> DeviceInput {
        DeviceInput::F64(ptr)
    }
}

impl Real for f32 {
    const PRECISION: Precision = Precision::Float32;
    const ZERO: Self = 0.0;
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    fn device_ptr(ptr: *const Self) -> DeviceInput {
        DeviceInput::F32(ptr)
    }
}

/// Element type of data already on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceDtype {
    F64,
    F32,
    /// Basis-state indices as `int64` / `usize`.
    I64,
}

impl DeviceDtype {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F64 => "float64",
            Self::F32 => "float32",
            Self::I64 => "int64",
        }
    }
}

/// A borrowed, typed pointer to a batch on the device.
#[derive(Clone, Copy, Debug)]
pub enum DeviceInput {
    F64(*const f64),
    F32(*const f32),
    I64(*const usize),
}

impl DeviceInput {
    pub fn dtype(self) -> DeviceDtype {
        match self {
            Self::F64(_) => DeviceDtype::F64,
            Self::F32(_) => DeviceDtype::F32,
            Self::I64(_) => DeviceDtype::I64,
        }
    }

    pub fn as_void(self) -> *const c_void {
        match self {
            Self::F64(p) => p as *const c_void,
            Self::F32(p) => p as *const c_void,
            Self::I64(p) => p as *const c_void,
        }
    }
}

/// Host data to encode.
#[derive(Clone, Copy, Debug)]
pub enum HostInput<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
}

impl HostInput<'_> {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::F64(s) => s.len(),
            Self::F32(s) => s.len(),
        }
    }

    pub fn dtype(&self) -> DeviceDtype {
        match self {
            Self::F64(_) => DeviceDtype::F64,
            Self::F32(_) => DeviceDtype::F32,
        }
    }

    /// Index of the first non-finite value, if any.
    pub fn first_non_finite(&self) -> Option<usize> {
        match self {
            Self::F64(s) => s.iter().position(|v| !v.is_finite()),
            Self::F32(s) => s.iter().position(|v| !v.is_finite()),
        }
    }

    pub fn get_f64(&self, i: usize) -> f64 {
        match self {
            Self::F64(s) => s[i],
            Self::F32(s) => s[i] as f64,
        }
    }
}

/// Where the input lives.
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    /// Data in host memory; the engine uploads it.
    Host(HostInput<'a>),
    /// Data already on this engine's device. `stream` is the CUDA stream the
    /// caller produced the data on (null for the default stream); the kernel
    /// runs on that stream and the call returns once it has completed.
    Device {
        ptr: DeviceInput,
        stream: *mut c_void,
    },
}

/// Batch geometry: `num_samples` samples of `sample_size` elements each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub num_samples: usize,
    pub sample_size: usize,
}

impl Shape {
    pub const fn new(num_samples: usize, sample_size: usize) -> Self {
        Self {
            num_samples,
            sample_size,
        }
    }

    pub fn total(&self) -> Result<usize> {
        self.num_samples
            .checked_mul(self.sample_size)
            .ok_or_else(|| {
                MahoutError::InvalidInput(format!(
                    "Batch size overflow: {} samples * {} elements",
                    self.num_samples, self.sample_size
                ))
            })
    }
}

/// Destination for `num_samples * 2^n` complex amplitudes.
#[derive(Clone, Copy, Debug)]
pub struct Output {
    ptr: *mut c_void,
    pub precision: Precision,
    pub state_len: usize,
}

impl Output {
    pub fn of(state: &GpuStateVector) -> Self {
        Self {
            ptr: state.ptr_void(),
            precision: state.precision(),
            state_len: 1 << state.num_qubits(),
        }
    }

    /// The same destination shifted forward by `samples` whole samples.
    pub fn offset_samples(self, samples: usize) -> Result<Self> {
        let elements = samples.checked_mul(self.state_len).ok_or_else(|| {
            MahoutError::MemoryAllocation(format!(
                "Output offset overflow: {} samples * {} elements",
                samples, self.state_len
            ))
        })?;
        let bytes = elements * self.precision.complex_bytes();
        Ok(Self {
            // SAFETY: the caller keeps offsets inside the allocation this
            // Output was created from.
            ptr: unsafe { self.ptr.cast::<u8>().add(bytes).cast::<c_void>() },
            ..self
        })
    }

    pub fn as_f64(&self) -> Result<*mut CuDoubleComplex> {
        match self.precision {
            Precision::Float64 => Ok(self.ptr as *mut CuDoubleComplex),
            Precision::Float32 => Err(precision_mismatch(self.precision, Precision::Float64)),
        }
    }

    pub fn as_f32(&self) -> Result<*mut CuComplex> {
        match self.precision {
            Precision::Float32 => Ok(self.ptr as *mut CuComplex),
            Precision::Float64 => Err(precision_mismatch(self.precision, Precision::Float32)),
        }
    }

    pub fn as_void(&self) -> *mut c_void {
        self.ptr
    }
}

fn precision_mismatch(have: Precision, want: Precision) -> MahoutError {
    MahoutError::InvalidInput(format!(
        "State vector precision mismatch (expected {:?} buffer, got {:?})",
        want, have
    ))
}

impl Precision {
    pub(crate) const fn complex_bytes(self) -> usize {
        match self {
            Self::Float32 => std::mem::size_of::<CuComplex>(),
            Self::Float64 => std::mem::size_of::<CuDoubleComplex>(),
        }
    }
}

// ---------------------------------------------------------------------------
// Launch context
// ---------------------------------------------------------------------------

/// The device and stream a kernel launches on.
pub struct LaunchCtx<'a> {
    pub device: &'a Arc<CudaDevice>,
    /// Raw `cudaStream_t`; null is the default stream.
    pub stream: *mut c_void,
}

impl<'a> LaunchCtx<'a> {
    pub fn new(device: &'a Arc<CudaDevice>, stream: *mut c_void) -> Self {
        Self { device, stream }
    }

    pub fn default_stream(device: &'a Arc<CudaDevice>) -> Self {
        Self::new(device, std::ptr::null_mut())
    }

    /// Resolve `module::name` and launch it on this context's stream.
    ///
    /// # Safety
    /// See [`qdp_kernels::launch`]: `args` must match the kernel's parameter
    /// list and every device pointer must be valid on the stream.
    pub unsafe fn launch(
        &self,
        module: &'static str,
        name: &'static str,
        cfg: LaunchConfig,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        let f =
            qdp_kernels::function(self.device, module, name).map_err(|e| kernel_err(name, e))?;
        // SAFETY: forwarded from the caller's contract.
        unsafe { qdp_kernels::launch(self.device, f, cfg, self.stream, args) }
            .map_err(|e| kernel_err(name, e))
    }

    /// Wait for everything queued on this context's stream.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: the stream is either null (default) or a live stream owned
        // by the caller for the duration of the encode call.
        unsafe { cudarc::driver::result::stream::synchronize(self.stream as _) }
            .map_err(|e| MahoutError::Cuda(format!("CUDA stream synchronize failed: {e}")))
    }
}

fn kernel_err(name: &str, e: KernelError) -> MahoutError {
    match e {
        KernelError::Unavailable => MahoutError::Cuda(e.to_string()),
        KernelError::Driver(_) => MahoutError::KernelLaunch(format!("{name}: {e}")),
        other => MahoutError::KernelLaunch(format!("{name}: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Deferred work
// ---------------------------------------------------------------------------

/// What a launch leaves behind to be settled once its stream has completed:
/// device buffers the queued kernels still read, and checks that need a
/// result copied back from the device (a validation flag, a norm).
///
/// Launches stay asynchronous; the caller synchronises the stream once and
/// then calls [`Pending::finish`].
/// A deferred check: runs after the stream is synchronised, may copy from the device.
type Check = Box<dyn FnOnce(&Arc<CudaDevice>) -> Result<()> + Send>;

#[derive(Default)]
pub struct Pending {
    keep: Vec<Box<dyn Any + Send>>,
    checks: Vec<Check>,
}

impl Pending {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep `buffer` alive until [`Pending::finish`].
    pub fn keep<T: Send + 'static>(&mut self, buffer: T) {
        self.keep.push(Box::new(buffer));
    }

    /// Run `check` after the stream has completed; it may copy from the device.
    pub fn check(&mut self, check: impl FnOnce(&Arc<CudaDevice>) -> Result<()> + Send + 'static) {
        self.checks.push(Box::new(check));
    }

    pub fn merge(&mut self, mut other: Pending) {
        self.keep.append(&mut other.keep);
        self.checks.append(&mut other.checks);
    }

    /// Drop the deferred checks, keeping only the buffer lifetimes. Used by
    /// the streaming path, which turns invalid samples into zero rows instead
    /// of failing the whole file.
    pub fn without_checks(mut self) -> Self {
        self.checks.clear();
        self
    }

    /// Run the checks in order, then release the buffers. Call only after the
    /// stream the launches went to has been synchronised.
    pub fn finish(self, device: &Arc<CudaDevice>) -> Result<()> {
        for check in self.checks {
            check(device)?;
        }
        drop(self.keep);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The Kernel trait
// ---------------------------------------------------------------------------

/// One encoding, implemented by one `.cu` file and one descriptor.
pub trait Kernel: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    /// Elements per sample for `num_qubits`.
    fn sample_size(&self, num_qubits: usize) -> usize;

    /// Which input element types have a device kernel.
    fn supports(&self, dtype: DeviceDtype) -> bool;

    /// Reject batch geometry this encoding cannot take. The default requires
    /// exactly [`Kernel::sample_size`] elements per sample.
    fn validate_shape(&self, shape: Shape, num_qubits: usize) -> Result<()> {
        let expected = self.sample_size(num_qubits);
        if shape.sample_size != expected {
            return Err(MahoutError::InvalidInput(format!(
                "{} encoding expects {} values per sample (sample_size={}) for {} qubits, got {}",
                self.display_name(),
                expected,
                expected,
                num_qubits,
                shape.sample_size
            )));
        }
        Ok(())
    }

    /// What one input element is called in error messages ("angle", "phase").
    fn element_noun(&self) -> &'static str {
        "element"
    }

    /// How the encoding is spelled at the start of an error message.
    fn display_name(&self) -> String {
        capitalised(self.name())
    }

    /// CPU-side check of host data before upload. The default rejects
    /// non-finite values.
    fn validate_host(&self, host: &HostInput, shape: Shape, _num_qubits: usize) -> Result<()> {
        if let Some(i) = host.first_non_finite() {
            return Err(MahoutError::InvalidInput(format!(
                "Sample {} {} {} must be finite, got {}",
                i / shape.sample_size,
                self.element_noun(),
                i % shape.sample_size,
                host.get_f64(i)
            )));
        }
        Ok(())
    }

    /// Device-side check of data the caller placed on the GPU. The default
    /// queues a finite-value check; its verdict is read in [`Pending::finish`].
    ///
    /// # Safety
    /// `input` must point at `shape.total()` elements valid on `ctx.stream`.
    unsafe fn validate_device(
        &self,
        ctx: &LaunchCtx,
        input: DeviceInput,
        shape: Shape,
        _num_qubits: usize,
    ) -> Result<Pending> {
        let total = shape.total()?;
        match input {
            // SAFETY: forwarded from the caller's contract.
            DeviceInput::F64(p) => unsafe {
                crate::gpu::validation::check_all_finite::<f64>(ctx, p, total, self.name())
            },
            DeviceInput::F32(p) => unsafe {
                crate::gpu::validation::check_all_finite::<f32>(ctx, p, total, self.name())
            },
            DeviceInput::I64(_) => Err(MahoutError::InvalidInput(format!(
                "{} encoding does not accept int64 input",
                self.name()
            ))),
        }
    }

    /// Precision of the state vector this kernel writes for `input`.
    fn output_precision(&self, input: DeviceDtype) -> Result<Precision> {
        if !self.supports(input) {
            return Err(MahoutError::NotImplemented(format!(
                "{} encoding has no {} kernel",
                self.name(),
                input.as_str()
            )));
        }
        Ok(match input {
            DeviceDtype::F32 => Precision::Float32,
            DeviceDtype::F64 | DeviceDtype::I64 => Precision::Float64,
        })
    }

    /// Queue the kernels that encode `shape.num_samples` samples resident at
    /// `input` into `out`, on `ctx.stream`, without waiting for them.
    ///
    /// Buffers the kernels read and checks that need device results go into
    /// the returned [`Pending`]; the caller synchronises the stream and then
    /// settles it. `out.precision` is what [`Kernel::output_precision`]
    /// returned for `input.dtype()`.
    ///
    /// # Safety
    /// `input` must hold `shape.total()` elements and `out` must have room for
    /// `shape.num_samples * out.state_len` amplitudes, both valid on the
    /// stream until it completes.
    unsafe fn launch(
        &self,
        ctx: &LaunchCtx,
        input: DeviceInput,
        shape: Shape,
        num_qubits: usize,
        out: Output,
    ) -> Result<Pending>;

    /// Upload host data and encode it. The default streams inputs at or above
    /// [`ASYNC_UPLOAD_THRESHOLD_BYTES`] through the pinned dual-stream
    /// pipeline, one whole-sample chunk at a time, and copies smaller inputs
    /// synchronously. Override only when an encoding has a better host-side
    /// strategy (amplitude does, for one huge sample).
    fn encode_host(
        &self,
        device: &Arc<CudaDevice>,
        host: HostInput,
        shape: Shape,
        num_qubits: usize,
        out: Output,
    ) -> Result<()> {
        match host {
            HostInput::F64(s) => default_encode_host(self, device, s, shape, num_qubits, out),
            HostInput::F32(s) => default_encode_host(self, device, s, shape, num_qubits, out),
        }
    }
}

/// Shared host-upload strategy; see [`Kernel::encode_host`].
pub fn default_encode_host<T: Real, K: Kernel + ?Sized>(
    kernel: &K,
    device: &Arc<CudaDevice>,
    host: &[T],
    shape: Shape,
    num_qubits: usize,
    out: Output,
) -> Result<()> {
    let bytes = std::mem::size_of_val(host);
    if bytes >= ASYNC_UPLOAD_THRESHOLD_BYTES && shape.num_samples > 1 {
        return chunked_encode_host(kernel, device, host, shape, num_qubits, out);
    }
    sync_encode_host(kernel, device, host, shape, num_qubits, out)
}

/// One synchronous upload, one launch on the default stream, one wait.
pub(crate) fn sync_encode_host<T: Real, K: Kernel + ?Sized>(
    kernel: &K,
    device: &Arc<CudaDevice>,
    host: &[T],
    shape: Shape,
    num_qubits: usize,
    out: Output,
) -> Result<()> {
    let bytes = std::mem::size_of_val(host);
    let input = {
        crate::profile_scope!("GPU::H2D_Input");
        device.htod_sync_copy(host).map_err(|e| {
            crate::gpu::memory::map_allocation_error(bytes, "input upload", Some(num_qubits), e)
        })?
    };
    let ctx = LaunchCtx::default_stream(device);
    let pending = {
        crate::profile_scope!("GPU::KernelLaunch");
        // SAFETY: `input` holds exactly `shape.total()` elements and stays
        // alive past the synchronize below.
        unsafe {
            kernel.launch(
                &ctx,
                T::device_ptr(*input.device_ptr() as *const T),
                shape,
                num_qubits,
                out,
            )?
        }
    };
    {
        crate::profile_scope!("GPU::Synchronize");
        device
            .synchronize()
            .map_err(|e| MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e)))?;
    }
    pending.finish(device)
}

#[cfg(target_os = "linux")]
fn chunked_encode_host<T: Real, K: Kernel + ?Sized>(
    kernel: &K,
    device: &Arc<CudaDevice>,
    host: &[T],
    shape: Shape,
    num_qubits: usize,
    out: Output,
) -> Result<()> {
    let sample_size = shape.sample_size;
    let pending = std::cell::RefCell::new(Pending::new());
    crate::gpu::pipeline::run_dual_stream_pipeline_aligned_typed::<T, _>(
        device,
        host,
        sample_size,
        |stream, input_ptr, chunk_offset, chunk_len| {
            let chunk = Shape::new(chunk_len / sample_size, sample_size);
            let ctx = LaunchCtx::new(device, stream.stream as *mut c_void);
            // SAFETY: the pipeline hands us a device buffer holding exactly
            // `chunk_len` elements, aligned to whole samples, valid on `stream`.
            let p = unsafe {
                kernel.launch(
                    &ctx,
                    T::device_ptr(input_ptr),
                    chunk,
                    num_qubits,
                    out.offset_samples(chunk_offset / sample_size)?,
                )?
            };
            pending.borrow_mut().merge(p);
            Ok(())
        },
    )?;
    // The pipeline has waited for its compute stream by the time it returns.
    pending.into_inner().finish(device)
}

#[cfg(not(target_os = "linux"))]
fn chunked_encode_host<T: Real, K: Kernel + ?Sized>(
    _kernel: &K,
    _device: &Arc<CudaDevice>,
    _host: &[T],
    _shape: Shape,
    _num_qubits: usize,
    _out: Output,
) -> Result<()> {
    Err(MahoutError::Cuda(
        "CUDA unavailable (non-Linux stub)".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// Encode `input` with `kernel` into a freshly allocated state vector.
///
/// This is the only path from data to amplitudes: host or device, `f32` or
/// `f64`, one sample or a batch.
pub fn encode(
    device: &Arc<CudaDevice>,
    kernel: &dyn Kernel,
    input: Input,
    shape: Shape,
    num_qubits: usize,
) -> Result<GpuStateVector> {
    validate_qubit_count(num_qubits)?;
    if shape.num_samples == 0 {
        return Err(MahoutError::InvalidInput(
            "Number of samples cannot be zero (num_samples must be greater than 0)".into(),
        ));
    }
    if shape.sample_size == 0 {
        return Err(MahoutError::InvalidInput(
            "Input data cannot be empty (sample_size = 0)".into(),
        ));
    }
    kernel.validate_shape(shape, num_qubits)?;

    match input {
        Input::Host(host) => {
            let expected = shape.total()?;
            if host.len() != expected {
                return Err(MahoutError::InvalidInput(format!(
                    "Input length {} doesn't match num_samples {} * sample_size {}",
                    host.len(),
                    shape.num_samples,
                    shape.sample_size
                )));
            }
            kernel.validate_host(&host, shape, num_qubits)?;
            let precision = kernel.output_precision(host.dtype())?;
            let state = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new_batch(device, shape.num_samples, num_qubits, precision)?
            };
            kernel.encode_host(device, host, shape, num_qubits, Output::of(&state))?;
            Ok(state)
        }
        Input::Device { ptr, stream } => {
            validate_cuda_input_ptr(device, ptr.as_void())?;
            let precision = kernel.output_precision(ptr.dtype())?;
            let ctx = LaunchCtx::new(device, stream);
            // SAFETY: the caller of `encode` guarantees `ptr` holds
            // `shape.total()` elements valid on `stream`.
            let mut pending = unsafe { kernel.validate_device(&ctx, ptr, shape, num_qubits)? };
            let state = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new_batch(device, shape.num_samples, num_qubits, precision)?
            };
            {
                crate::profile_scope!("GPU::KernelLaunch");
                // SAFETY: as above; `state` is freshly allocated for this batch.
                pending.merge(unsafe {
                    kernel.launch(&ctx, ptr, shape, num_qubits, Output::of(&state))?
                });
            }
            {
                crate::profile_scope!("GPU::Synchronize");
                ctx.synchronize()?;
            }
            pending.finish(device)?;
            Ok(state)
        }
    }
}

/// Reject pointers that are null, not device memory, or on another device.
#[cfg(target_os = "linux")]
fn validate_cuda_input_ptr(device: &CudaDevice, ptr: *const c_void) -> Result<()> {
    use crate::gpu::cuda_ffi::{
        CUDA_MEMORY_TYPE_DEVICE, CUDA_MEMORY_TYPE_MANAGED, CudaPointerAttributes,
        cudaPointerGetAttributes,
    };
    if ptr.is_null() {
        return Err(MahoutError::InvalidInput(
            "Input GPU pointer is null".to_string(),
        ));
    }
    let mut attrs = CudaPointerAttributes {
        memory_type: 0,
        device: 0,
        device_pointer: std::ptr::null_mut(),
        host_pointer: std::ptr::null_mut(),
        is_managed: 0,
        allocation_flags: 0,
    };
    // SAFETY: `attrs` is a valid out-parameter; the call only inspects `ptr`.
    let ret = unsafe { cudaPointerGetAttributes(&mut attrs as *mut _, ptr) };
    if ret != 0 {
        return Err(MahoutError::InvalidInput(format!(
            "cudaPointerGetAttributes failed for input pointer: {} ({})",
            ret,
            crate::error::cuda_error_to_string(ret)
        )));
    }
    if attrs.memory_type != CUDA_MEMORY_TYPE_DEVICE && attrs.memory_type != CUDA_MEMORY_TYPE_MANAGED
    {
        return Err(MahoutError::InvalidInput(format!(
            "Input pointer is not CUDA device memory (memory_type={})",
            attrs.memory_type
        )));
    }
    let device_ordinal = device.ordinal() as i32;
    if attrs.device >= 0 && attrs.device != device_ordinal {
        return Err(MahoutError::InvalidInput(format!(
            "Input pointer device mismatch: pointer on cuda:{}, engine on cuda:{}",
            attrs.device, device_ordinal
        )));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn validate_cuda_input_ptr(_device: &CudaDevice, _ptr: *const c_void) -> Result<()> {
    Err(MahoutError::Cuda(
        "CUDA unavailable (non-Linux stub)".to_string(),
    ))
}

/// The text of an `InvalidInput` error without its "Invalid input:" prefix,
/// for wrapping in a more specific message.
pub(crate) fn invalid_input_text(e: &MahoutError) -> String {
    match e {
        MahoutError::InvalidInput(m) => m.clone(),
        other => other.to_string(),
    }
}

/// "angle" -> "Angle" for the start of an error message.
pub(crate) fn capitalised(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Error for an (input dtype, output precision) pair a kernel has no code for.
pub(crate) fn unsupported(name: &str, input: DeviceDtype, out: Precision) -> MahoutError {
    MahoutError::NotImplemented(format!(
        "{} encoding has no kernel for {} input producing {:?} output",
        name,
        input.as_str(),
        out
    ))
}

/// Choose the `f64` or `f32` variant of a kernel symbol by output precision.
pub(crate) fn symbol(
    precision: Precision,
    f64_name: &'static str,
    f32_name: &'static str,
) -> &'static str {
    match precision {
        Precision::Float64 => f64_name,
        Precision::Float32 => f32_name,
    }
}
