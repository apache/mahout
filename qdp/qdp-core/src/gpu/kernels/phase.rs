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

//! Phase encoding: `|psi(x)> = ⊗_k (1/√2)(|0> + e^{i x_k}|1>)`.
//!
//! Device code: `qdp-kernels/src/phase.cu`.

use qdp_kernels::{LaunchConfig, config, kernel_args};

use super::{DeviceDtype, DeviceInput, Kernel, LaunchCtx, Output, Pending, Shape, unsupported};
use crate::error::Result;
use crate::gpu::memory::Precision;

pub struct PhaseKernel;

impl Kernel for PhaseKernel {
    fn name(&self) -> &'static str {
        "phase"
    }

    fn sample_size(&self, num_qubits: usize) -> usize {
        num_qubits
    }

    fn element_noun(&self) -> &'static str {
        "phase"
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
        let state_len = out.state_len;
        let total = shape.num_samples.checked_mul(state_len).ok_or_else(|| {
            crate::error::MahoutError::InvalidInput("Phase batch output size overflow".into())
        })?;
        let cfg = LaunchConfig::grid_stride(total, config::MAX_GRID_BLOCKS);
        let norm_factor: f64 = std::f64::consts::FRAC_1_SQRT_2.powi(num_qubits as i32);
        let num_qubits = num_qubits as u32;

        // SAFETY: argument list matches `phase_encode_batch_kernel`.
        unsafe {
            match (input, out.precision) {
                (DeviceInput::F64(p), Precision::Float64) => ctx.launch(
                    "phase",
                    "phase_encode_batch_kernel",
                    cfg,
                    &mut kernel_args![
                        p,
                        out.as_f64()?,
                        shape.num_samples,
                        state_len,
                        num_qubits,
                        norm_factor
                    ],
                ),
                (p, prec) => Err(unsupported(self.name(), p.dtype(), prec)),
            }
        }
        .map(|_| Pending::new())
    }
}
