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

//! Amplitude encoding: a sample of up to `2^n` reals becomes the normalised
//! amplitudes of an `n`-qubit state, zero-padded to the full length.
//!
//! Each launch queues a per-sample inverse-norm reduction, a finalize step
//! that turns sums into inverse norms and raises a flag for zero or
//! non-finite ones, and the encode kernel. Nothing waits; the flag is read
//! by the returned [`Pending`]. A single sample uses the wide single-sample
//! reduction, and a small single host sample takes its norm on the CPU.
//!
//! Device code: `qdp-kernels/src/amplitude.cu`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr as _, DevicePtrMut, sys};
use qdp_kernels::{CuDoubleComplex, LaunchConfig, config, kernel_args};

use super::{
    ASYNC_UPLOAD_THRESHOLD_BYTES, DeviceDtype, DeviceInput, HostInput, Kernel, LaunchCtx, Output,
    Pending, Real, Shape, symbol, sync_encode_host, unsupported,
};
use crate::error::{MahoutError, Result};
use crate::gpu::memory::Precision;
use crate::gpu::validation::{alloc_flag, read_flag};
use crate::preprocessing::Preprocessor;

/// Below this many elements a single host sample's norm is cheaper on the CPU
/// than a reduction kernel plus a flag round trip.
const CPU_NORM_THRESHOLD: usize = 4096;

pub struct AmplitudeKernel;

impl Kernel for AmplitudeKernel {
    fn name(&self) -> &'static str {
        "amplitude"
    }

    fn sample_size(&self, num_qubits: usize) -> usize {
        1 << num_qubits
    }

    fn supports(&self, dtype: DeviceDtype) -> bool {
        matches!(dtype, DeviceDtype::F64 | DeviceDtype::F32)
    }

    fn validate_shape(&self, shape: Shape, num_qubits: usize) -> Result<()> {
        let state_len = 1usize << num_qubits;
        if shape.sample_size > state_len {
            return Err(MahoutError::InvalidInput(format!(
                "Sample size {} exceeds state vector size {} (2^{} qubits)",
                shape.sample_size, state_len, num_qubits
            )));
        }
        Ok(())
    }

    fn validate_host(&self, _host: &HostInput, _shape: Shape, _num_qubits: usize) -> Result<()> {
        // The norm reduction rejects NaN, Inf and all-zero samples; a CPU pass
        // over up to 2^n values per sample would cost more than the encode.
        Ok(())
    }

    unsafe fn validate_device(
        &self,
        _ctx: &LaunchCtx,
        _input: DeviceInput,
        _shape: Shape,
        _num_qubits: usize,
    ) -> Result<Pending> {
        // As above: the norm reduction is the validation.
        Ok(Pending::new())
    }

    unsafe fn launch(
        &self,
        ctx: &LaunchCtx,
        input: DeviceInput,
        shape: Shape,
        _num_qubits: usize,
        out: Output,
    ) -> Result<Pending> {
        // SAFETY: forwarded from the caller's contract.
        unsafe {
            match (input, out.precision) {
                (DeviceInput::F64(p), Precision::Float64) => {
                    launch_typed::<f64>(ctx, p, shape, out.as_f64()? as *mut c_void, out.state_len)
                }
                (DeviceInput::F32(p), Precision::Float32) => {
                    launch_typed::<f32>(ctx, p, shape, out.as_f32()? as *mut c_void, out.state_len)
                }
                (p, prec) => Err(unsupported(self.name(), p.dtype(), prec)),
            }
        }
    }

    fn encode_host(
        &self,
        device: &Arc<CudaDevice>,
        host: HostInput,
        shape: Shape,
        num_qubits: usize,
        out: Output,
    ) -> Result<()> {
        if let HostInput::F64(data) = host
            && shape.num_samples == 1
        {
            if std::mem::size_of_val(data) >= ASYNC_UPLOAD_THRESHOLD_BYTES {
                return encode_single_streamed(device, data, num_qubits, out);
            }
            if data.len() < CPU_NORM_THRESHOLD {
                return encode_single_small(device, data, out);
            }
        }
        // Batches upload in one copy: the per-sample norm reduction needs
        // every sample whole.
        match host {
            HostInput::F64(s) => sync_encode_host(self, device, s, shape, num_qubits, out),
            HostInput::F32(s) => sync_encode_host(self, device, s, shape, num_qubits, out),
        }
    }
}

/// Queue the per-sample inverse-norm reduction for a device batch.
///
/// Returns the norm buffer (read by the encode kernel) and the flag the
/// finalize kernel raises for zero or non-finite norms. Neither is valid
/// until the stream reaches the kernels.
///
/// # Safety
/// `input` must hold `shape.total()` elements valid on `ctx.stream`.
pub unsafe fn inv_norms<T: Real>(
    ctx: &LaunchCtx,
    input: *const T,
    shape: Shape,
) -> Result<(CudaSlice<T>, CudaSlice<i32>)> {
    crate::profile_scope!("GPU::BatchNormKernel");
    let num_samples = shape.num_samples;
    let sample_size = shape.sample_size;
    let device = ctx.device;

    // SAFETY: freshly allocated and zeroed on the launch stream before use.
    let mut norms = unsafe { device.alloc::<T>(num_samples) }.map_err(|e| {
        MahoutError::MemoryAllocation(format!("Failed to allocate norm buffer: {:?}", e))
    })?;
    let norms_ptr = *norms.device_ptr_mut();
    unsafe {
        cudarc::driver::result::memset_d8_async(
            norms_ptr,
            0,
            num_samples * std::mem::size_of::<T>(),
            ctx.stream as sys::CUstream,
        )
        .map_err(|e| MahoutError::Cuda(format!("Failed to zero norm buffer: {e}")))?;
    }
    let norms_typed = norms_ptr as *mut T;
    let block = config::DEFAULT_BLOCK_SIZE;
    let elements_per_block = block * 2;

    // SAFETY: argument lists match the named reduction kernels.
    unsafe {
        if num_samples == 1 {
            // One sample: spread the reduction over the whole GPU.
            ctx.launch(
                "amplitude",
                symbol(T::PRECISION, "l2_norm_kernel", "l2_norm_kernel_f32"),
                LaunchConfig::grid_stride(sample_size.div_ceil(2), config::MAX_GRID_BLOCKS_L2_NORM),
                &mut kernel_args![input, sample_size, norms_typed],
            )?;
        } else {
            // Geometry mirrors the batch reduction kernel: two elements per
            // thread, up to MAX_BLOCKS_PER_SAMPLE blocks per sample, never
            // more blocks than the device's grid limit.
            let max_grid = device
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
                .map(|v| v.max(1) as usize)
                .unwrap_or(config::CUDA_MAX_GRID_DIM_1D);
            if num_samples > max_grid {
                return Err(MahoutError::InvalidInput(format!(
                    "Batch of {} samples exceeds the device grid limit {}",
                    num_samples, max_grid
                )));
            }
            let mut blocks_per_sample = sample_size
                .div_ceil(elements_per_block)
                .clamp(1, config::MAX_BLOCKS_PER_SAMPLE);
            if num_samples * blocks_per_sample > max_grid {
                blocks_per_sample = (max_grid / num_samples).max(1);
            }
            ctx.launch(
                "amplitude",
                symbol(
                    T::PRECISION,
                    "l2_norm_batch_kernel",
                    "l2_norm_batch_kernel_f32",
                ),
                LaunchConfig::exact((num_samples * blocks_per_sample) as u32, block as u32),
                &mut kernel_args![
                    input,
                    num_samples,
                    sample_size,
                    blocks_per_sample,
                    norms_typed
                ],
            )?;
        }
    }

    let mut flag = alloc_flag(ctx, "norm")?;
    let flag_ptr = *flag.device_ptr_mut() as *mut i32;
    // SAFETY: argument list matches `finalize_inv_norm_kernel[_f32]`.
    unsafe {
        ctx.launch(
            "amplitude",
            symbol(
                T::PRECISION,
                "finalize_inv_norm_kernel",
                "finalize_inv_norm_kernel_f32",
            ),
            LaunchConfig::grid_1d(num_samples),
            &mut kernel_args![norms_typed, num_samples, flag_ptr],
        )?;
    }
    Ok((norms, flag))
}

/// # Safety
/// `input` holds `shape.total()` elements and `state` has room for
/// `shape.num_samples * state_len` amplitudes of `T`'s precision.
unsafe fn launch_typed<T: Real>(
    ctx: &LaunchCtx,
    input: *const T,
    shape: Shape,
    state: *mut c_void,
    state_len: usize,
) -> Result<Pending> {
    // SAFETY: forwarded from the caller's contract.
    let (norms, flag) = unsafe { inv_norms::<T>(ctx, input, shape)? };
    let norms_ptr = *norms.device_ptr() as *const T;
    let num_samples = shape.num_samples;
    let sample_size = shape.sample_size;
    let work = num_samples * (state_len / 2);

    {
        crate::profile_scope!("GPU::BatchKernelLaunch");
        // SAFETY: argument list matches `amplitude_encode_batch_kernel[_f32]`.
        unsafe {
            ctx.launch(
                "amplitude",
                symbol(
                    T::PRECISION,
                    "amplitude_encode_batch_kernel",
                    "amplitude_encode_batch_kernel_f32",
                ),
                LaunchConfig::grid_stride(work, config::MAX_GRID_BLOCKS),
                &mut kernel_args![input, state, norms_ptr, num_samples, sample_size, state_len],
            )?;
        }
    }
    let mut pending = Pending::new();
    pending.keep(norms);
    pending.check(move |device| {
        if read_flag(device, &flag, "norm")? != 0 {
            return Err(MahoutError::InvalidInput(
                "Input data has zero norm or non-finite values: one or more samples have zero \
                 or non-finite norm (NaN, Inf, or all zeros)"
                    .to_string(),
            ));
        }
        Ok(())
    });
    Ok(pending)
}

/// One small host sample: norm on the CPU, one upload, one kernel, one wait.
fn encode_single_small(device: &Arc<CudaDevice>, data: &[f64], out: Output) -> Result<()> {
    let inv_norm = 1.0 / Preprocessor::calculate_l2_norm(data)?;
    let input = device.htod_sync_copy(data).map_err(|e| {
        crate::gpu::memory::map_allocation_error(
            std::mem::size_of_val(data),
            "input upload",
            None,
            e,
        )
    })?;
    let ctx = LaunchCtx::default_stream(device);
    let input_len = data.len();
    let state_len = out.state_len;
    let state = out.as_f64()?;
    // SAFETY: argument list matches `amplitude_encode_kernel`; `input` lives
    // past the synchronize below.
    unsafe {
        ctx.launch(
            "amplitude",
            "amplitude_encode_kernel",
            LaunchConfig::grid_1d(state_len.div_ceil(2)),
            &mut kernel_args![
                *input.device_ptr() as *const f64,
                state,
                input_len,
                state_len,
                inv_norm
            ],
        )?;
    }
    device
        .synchronize()
        .map_err(|e| MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e)))
}

/// One sample too large for a single synchronous copy: norm on the CPU,
/// stream the data through the dual-stream pipeline, zero the padding tail.
#[cfg(target_os = "linux")]
fn encode_single_streamed(
    device: &Arc<CudaDevice>,
    data: &[f64],
    num_qubits: usize,
    out: Output,
) -> Result<()> {
    let state_len = 1usize << num_qubits;
    let inv_norm = 1.0 / Preprocessor::calculate_l2_norm(data)?;
    let base = out.as_f64()?;

    crate::gpu::pipeline::run_dual_stream_pipeline(
        device,
        data,
        |stream, input_ptr, chunk_offset, chunk_len| {
            // SAFETY: `chunk_offset + chunk_len <= data.len() <= state_len`.
            let chunk_state = unsafe { base.add(chunk_offset) };
            let ctx = LaunchCtx::new(device, stream.stream as *mut c_void);
            // Each chunk is encoded as its own (input_len == state_len) span
            // so writes stay inside the chunk; padding is handled below.
            // SAFETY: argument list matches `amplitude_encode_kernel`.
            unsafe {
                ctx.launch(
                    "amplitude",
                    "amplitude_encode_kernel",
                    LaunchConfig::grid_1d(chunk_len.div_ceil(2)),
                    &mut kernel_args![input_ptr, chunk_state, chunk_len, chunk_len, inv_norm],
                )
            }
        },
    )?;

    if data.len() < state_len {
        let padding = state_len - data.len();
        // SAFETY: the tail lies inside the state allocation.
        let tail = unsafe { base.add(data.len()) } as sys::CUdeviceptr;
        unsafe {
            cudarc::driver::result::memset_d8_async(
                tail,
                0,
                padding * std::mem::size_of::<CuDoubleComplex>(),
                std::ptr::null_mut(),
            )
            .map_err(|e| MahoutError::Cuda(format!("Failed to zero-fill padding region: {e}")))?;
        }
        device
            .synchronize()
            .map_err(|e| MahoutError::Cuda(format!("Failed to sync after padding: {:?}", e)))?;
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn encode_single_streamed(
    _device: &Arc<CudaDevice>,
    _data: &[f64],
    _num_qubits: usize,
    _out: Output,
) -> Result<()> {
    Err(MahoutError::Cuda(
        "CUDA unavailable (non-Linux stub)".to_string(),
    ))
}
