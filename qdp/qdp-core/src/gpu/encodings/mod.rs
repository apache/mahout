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

// Quantum encoding strategies (Strategy Pattern)

use std::sync::Arc;
use arrow::array::Float64Array;
use cudarc::driver::CudaDevice;
use crate::error::Result;
use crate::gpu::memory::GpuStateVector;
use crate::gpu::pool::StagingBufferPool;
use crate::preprocessing::Preprocessor;

/// Quantum encoding strategy interface
/// Implemented by: AmplitudeEncoder, AngleEncoder, BasisEncoder
pub trait QuantumEncoder: Send + Sync {
    /// Encode classical data to quantum state on GPU
    fn encode(
        &self,
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector>;

    /// Encode from chunked Arrow arrays
    ///
    /// Default implementation flattens chunks. (TODO: Encoders can override for true zero-copy.)
    fn encode_chunked(
        &self,
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
        chunks: &[Float64Array],
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        // Default: flatten and use regular encode
        let data = crate::io::arrow_to_vec_chunked(chunks);
        self.encode(device, pool, &data, num_qubits)
    }

    /// Validate input data before encoding
    fn validate_input(&self, data: &[f64], num_qubits: usize) -> Result<()> {
        Preprocessor::validate_input(data, num_qubits)
    }

    /// Strategy name
    fn name(&self) -> &'static str;

    /// Strategy description
    fn description(&self) -> &'static str;
}

// Encoding implementations
pub mod amplitude;
pub mod angle;
pub mod basis;

pub use amplitude::AmplitudeEncoder;
pub use angle::AngleEncoder;
pub use basis::BasisEncoder;

/// Create encoder by name: "amplitude", "angle", or "basis"
pub fn get_encoder(name: &str) -> Result<Box<dyn QuantumEncoder>> {
    match name.to_lowercase().as_str() {
        "amplitude" => Ok(Box::new(AmplitudeEncoder)),
        "angle" => Ok(Box::new(AngleEncoder)),
        "basis" => Ok(Box::new(BasisEncoder)),
        _ => Err(crate::error::MahoutError::InvalidInput(
            format!("Unknown encoder: {}. Available: amplitude, angle, basis", name)
        )),
    }
}
