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

// Basis encoding (placeholder)
// TODO: Map integers to computational basis states

use std::sync::Arc;
use cudarc::driver::CudaDevice;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;
use crate::gpu::pool::StagingBufferPool;
use super::QuantumEncoder;

/// Basis encoding (not implemented)
/// TODO: Map integers to basis states (e.g., 3 → |011⟩)
pub struct BasisEncoder;

impl QuantumEncoder for BasisEncoder {
    fn encode(
        &self,
        _device: &Arc<CudaDevice>,
        _pool: &Arc<StagingBufferPool>,
        _data: &[f64],
        _num_qubits: usize,
    ) -> Result<GpuStateVector> {
        Err(MahoutError::InvalidInput(
            "Basis encoding not yet implemented. Use 'amplitude' encoding for now.".to_string()
        ))
    }

    fn name(&self) -> &'static str {
        "basis"
    }

    fn description(&self) -> &'static str {
        "Basis encoding (not implemented)"
    }
}
