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

//! IQP encoding: `|psi> = H^⊗n · U_phase(x) · H^⊗n |0>`.
//!
//! Small qubit counts use a direct kernel. Larger ones compute the phase
//! array and apply a fast Walsh-Hadamard transform; a single sample that
//! fits in shared memory does all of that in one fused launch.
//!
//! Device code: `qdp-kernels/src/iqp.cu`.

use qdp_kernels::{CuDoubleComplex, LaunchConfig, config, kernel_args};

use super::{DeviceDtype, DeviceInput, Kernel, LaunchCtx, Output, Pending, Shape, unsupported};
use crate::error::{MahoutError, Result};
use crate::gpu::memory::Precision;

pub struct IqpKernel {
    enable_zz: bool,
}

static IQP_FULL: IqpKernel = IqpKernel { enable_zz: true };
static IQP_Z: IqpKernel = IqpKernel { enable_zz: false };

/// IQP with single-qubit Z and two-qubit ZZ phases.
pub fn iqp_full_kernel() -> &'static IqpKernel {
    &IQP_FULL
}

/// IQP with single-qubit Z phases only.
pub fn iqp_z_kernel() -> &'static IqpKernel {
    &IQP_Z
}

impl Kernel for IqpKernel {
    fn name(&self) -> &'static str {
        if self.enable_zz { "iqp" } else { "iqp-z" }
    }

    fn sample_size(&self, num_qubits: usize) -> usize {
        if self.enable_zz {
            num_qubits + num_qubits * num_qubits.saturating_sub(1) / 2
        } else {
            num_qubits
        }
    }

    fn display_name(&self) -> String {
        if self.enable_zz { "IQP" } else { "IQP-Z" }.to_string()
    }

    fn element_noun(&self) -> &'static str {
        "parameter"
    }

    fn supports(&self, dtype: DeviceDtype) -> bool {
        matches!(dtype, DeviceDtype::F64)
    }

    unsafe fn launch(
        &self,
        ctx: &LaunchCtx,
        input: DeviceInput,
        shape: Shape,
        num_qubits: usize,
        out: Output,
    ) -> Result<Pending> {
        let (DeviceInput::F64(data), Precision::Float64) = (input, out.precision) else {
            return Err(unsupported(self.name(), input.dtype(), out.precision));
        };
        let state = out.as_f64()?;
        let state_len = out.state_len;
        let num_samples = shape.num_samples;
        let total = num_samples
            .checked_mul(state_len)
            .ok_or_else(|| MahoutError::InvalidInput("IQP batch output size overflow".into()))?;
        let norm_factor = 1.0 / state_len as f64;
        let enable_zz: i32 = if self.enable_zz { 1 } else { 0 };
        let data_len = shape.sample_size as u32;
        let n = num_qubits as u32;

        // SAFETY: each argument list matches the named kernel's signature.
        unsafe {
            if num_qubits < config::FWT_MIN_QUBITS {
                ctx.launch(
                    "iqp",
                    "iqp_encode_batch_kernel_naive",
                    LaunchConfig::grid_stride(total, config::MAX_GRID_BLOCKS),
                    &mut kernel_args![data, state, num_samples, state_len, n, data_len, enable_zz],
                )?;
                return Ok(Pending::new());
            }

            if num_samples == 1 && num_qubits <= config::FWT_SHARED_MEM_THRESHOLD {
                let shared = state_len * std::mem::size_of::<CuDoubleComplex>();
                ctx.launch(
                    "iqp",
                    "iqp_phase_fwt_shared_normalize_kernel",
                    LaunchConfig::exact(1, config::DEFAULT_BLOCK_SIZE as u32)
                        .with_shared_bytes(shared),
                    &mut kernel_args![data, state, state_len, n, enable_zz, norm_factor],
                )?;
                return Ok(Pending::new());
            }

            ctx.launch(
                "iqp",
                "iqp_phase_batch_kernel",
                LaunchConfig::grid_stride(total, config::MAX_GRID_BLOCKS),
                &mut kernel_args![
                    data,
                    state,
                    num_samples,
                    state_len,
                    n,
                    data_len,
                    enable_zz,
                    norm_factor
                ],
            )?;

            let pairs = num_samples * (state_len >> 1);
            let fwt_cfg = LaunchConfig::grid_stride(pairs, config::MAX_GRID_BLOCKS);
            for stage in 0..n {
                ctx.launch(
                    "iqp",
                    "fwt_butterfly_batch_kernel",
                    fwt_cfg,
                    &mut kernel_args![state, num_samples, state_len, n, stage],
                )?;
            }
        }
        Ok(Pending::new())
    }
}
