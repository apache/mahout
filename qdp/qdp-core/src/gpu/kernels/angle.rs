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

//! Angle encoding: `|psi(x)> = ⊗_k (cos(x_k)|0> + sin(x_k)|1>)`.
//!
//! Device code: `qdp-kernels/src/angle.cu`.

use qdp_kernels::{LaunchConfig, config, kernel_args};

use super::{
    DeviceDtype, DeviceInput, Kernel, LaunchCtx, Output, Pending, Shape, symbol, unsupported,
};
use crate::error::Result;
use crate::gpu::memory::Precision;

pub struct AngleKernel;

impl Kernel for AngleKernel {
    fn name(&self) -> &'static str {
        "angle"
    }

    fn sample_size(&self, num_qubits: usize) -> usize {
        num_qubits
    }

    fn element_noun(&self) -> &'static str {
        "angle"
    }

    fn supports(&self, dtype: DeviceDtype) -> bool {
        matches!(dtype, DeviceDtype::F64 | DeviceDtype::F32)
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
            crate::error::MahoutError::InvalidInput("Angle batch output size overflow".into())
        })?;
        let cfg = LaunchConfig::grid_stride(total, config::MAX_GRID_BLOCKS);
        let num_qubits = num_qubits as u32;
        let name = symbol(
            out.precision,
            "angle_encode_batch_kernel",
            "angle_encode_batch_kernel_f32",
        );

        // SAFETY: argument list matches `angle_encode_batch_kernel[_f32]`.
        unsafe {
            match (input, out.precision) {
                (DeviceInput::F64(p), Precision::Float64) => ctx.launch(
                    "angle",
                    name,
                    cfg,
                    &mut kernel_args![p, out.as_f64()?, shape.num_samples, state_len, num_qubits],
                ),
                (DeviceInput::F32(p), Precision::Float32) => ctx.launch(
                    "angle",
                    name,
                    cfg,
                    &mut kernel_args![p, out.as_f32()?, shape.num_samples, state_len, num_qubits],
                ),
                (p, prec) => Err(unsupported(self.name(), p.dtype(), prec)),
            }
        }
        .map(|_| Pending::new())
    }
}
