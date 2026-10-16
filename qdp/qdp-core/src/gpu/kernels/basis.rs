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

//! Basis encoding: one integer index per sample selects a computational
//! basis state.
//!
//! Indices may arrive as `int64` (used as-is after a queued range check), or
//! as `float32` / `float64` (validated and cast on the device). Host input is
//! validated and cast on the CPU, where the error message can name the
//! sample, and uploaded as `int64`.
//!
//! Device code: `qdp-kernels/src/basis.cu` and the index checks in
//! `qdp-kernels/src/validation.cu`.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, DevicePtr as _};
use qdp_kernels::{LaunchConfig, config, kernel_args};

use super::{
    DeviceDtype, DeviceInput, HostInput, Kernel, LaunchCtx, Output, Pending, Shape, symbol,
};
use crate::error::{MahoutError, Result};
use crate::gpu::memory::Precision;
use crate::gpu::validation;

pub struct BasisKernel;

fn validate_basis_index(value: f64, state_len: usize) -> std::result::Result<(), String> {
    if !value.is_finite() {
        return Err("Basis index must be a finite number".to_string());
    }
    if value < 0.0 {
        return Err(format!("Basis index must be non-negative, got {}", value));
    }
    if value.fract() != 0.0 {
        return Err(format!(
            "Basis index must be an integer, got {} (hint: use .round() if needed)",
            value
        ));
    }
    if value >= state_len as f64 {
        return Err(format!(
            "Basis index {} exceeds state vector size {} (must be < 2^num_qubits)",
            value, state_len
        ));
    }
    Ok(())
}

impl Kernel for BasisKernel {
    fn name(&self) -> &'static str {
        "basis"
    }

    fn sample_size(&self, _num_qubits: usize) -> usize {
        1
    }

    fn supports(&self, dtype: DeviceDtype) -> bool {
        matches!(
            dtype,
            DeviceDtype::F64 | DeviceDtype::F32 | DeviceDtype::I64
        )
    }

    fn validate_shape(&self, shape: Shape, _num_qubits: usize) -> Result<()> {
        if shape.sample_size != 1 {
            return Err(MahoutError::InvalidInput(format!(
                "Basis encoding expects exactly 1 value per sample (the basis index, sample_size=1), got {}",
                shape.sample_size
            )));
        }
        Ok(())
    }

    fn validate_host(&self, host: &HostInput, _shape: Shape, num_qubits: usize) -> Result<()> {
        let state_len = 1usize << num_qubits;
        for i in 0..host.len() {
            validate_basis_index(host.get_f64(i), state_len)
                .map_err(|e| MahoutError::InvalidInput(format!("Sample {}: {}", i, e)))?;
        }
        Ok(())
    }

    unsafe fn validate_device(
        &self,
        ctx: &LaunchCtx,
        input: DeviceInput,
        shape: Shape,
        num_qubits: usize,
    ) -> Result<Pending> {
        match input {
            // SAFETY: forwarded from the caller's contract.
            DeviceInput::I64(p) => unsafe {
                validation::check_basis_indices_in_range(
                    ctx,
                    p,
                    shape.num_samples,
                    1usize << num_qubits,
                )
            },
            // Float indices are validated by the cast kernel in `launch`.
            DeviceInput::F64(_) | DeviceInput::F32(_) => Ok(Pending::new()),
        }
    }

    unsafe fn launch(
        &self,
        ctx: &LaunchCtx,
        input: DeviceInput,
        shape: Shape,
        num_qubits: usize,
        out: Output,
    ) -> Result<Pending> {
        let state_len = out.state_len;
        let num_samples = shape.num_samples;
        let mut pending = Pending::new();

        // Float indices are cast into a device buffer that the encode kernel
        // reads; it stays alive in `pending` until the stream is done.
        // SAFETY: forwarded from the caller's contract.
        let indices: *const usize = unsafe {
            match input {
                DeviceInput::I64(p) => p,
                DeviceInput::F32(p) => {
                    let (buf, check) =
                        validation::cast_basis_indices::<f32>(ctx, p, num_samples, state_len)?;
                    let ptr = *buf.device_ptr() as *const usize;
                    pending.keep(buf);
                    pending.merge(check);
                    ptr
                }
                DeviceInput::F64(p) => {
                    let (buf, check) =
                        validation::cast_basis_indices::<f64>(ctx, p, num_samples, state_len)?;
                    let ptr = *buf.device_ptr() as *const usize;
                    pending.keep(buf);
                    pending.merge(check);
                    ptr
                }
            }
        };

        let cfg = LaunchConfig::grid_stride(num_samples * state_len, config::MAX_GRID_BLOCKS);
        let n = num_qubits as u32;
        let name = symbol(
            out.precision,
            "basis_encode_batch_kernel",
            "basis_encode_batch_kernel_f32",
        );
        // SAFETY: argument list matches `basis_encode_batch_kernel[_f32]`.
        unsafe {
            match out.precision {
                Precision::Float64 => ctx.launch(
                    "basis",
                    name,
                    cfg,
                    &mut kernel_args![indices, out.as_f64()?, num_samples, state_len, n],
                )?,
                Precision::Float32 => ctx.launch(
                    "basis",
                    name,
                    cfg,
                    &mut kernel_args![indices, out.as_f32()?, num_samples, state_len, n],
                )?,
            }
        }
        Ok(pending)
    }

    fn encode_host(
        &self,
        device: &Arc<CudaDevice>,
        host: HostInput,
        shape: Shape,
        num_qubits: usize,
        out: Output,
    ) -> Result<()> {
        // Host indices were validated on the CPU by `validate_host`, so cast
        // there too and upload `usize` directly: one copy, one kernel.
        let indices: Vec<usize> = (0..host.len()).map(|i| host.get_f64(i) as usize).collect();
        let indices_gpu = {
            crate::profile_scope!("GPU::H2D_Indices");
            device.htod_sync_copy(&indices).map_err(|e| {
                crate::gpu::memory::map_allocation_error(
                    indices.len() * std::mem::size_of::<usize>(),
                    "basis indices upload",
                    Some(num_qubits),
                    e,
                )
            })?
        };
        let ctx = LaunchCtx::default_stream(device);
        // SAFETY: `indices_gpu` holds `shape.num_samples` indices and outlives
        // the synchronize below.
        let pending = unsafe {
            self.launch(
                &ctx,
                DeviceInput::I64(*indices_gpu.device_ptr() as *const usize),
                shape,
                num_qubits,
                out,
            )?
        };
        device
            .synchronize()
            .map_err(|e| MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e)))?;
        pending.finish(device)
    }
}
