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

pub mod dlpack;
pub mod gpu;
pub mod error;
pub mod preprocessing;
pub mod io;

#[macro_use]
mod profiling;

pub use error::{MahoutError, Result};

use std::sync::Arc;
use arrow::array::Float64Array;
use cudarc::driver::CudaDevice;
use crate::dlpack::DLManagedTensor;
use crate::gpu::{get_encoder, StagingBufferPool};

/// Main entry point for Mahout QDP
///
/// Manages GPU context and dispatches encoding tasks.
/// Provides unified interface for device management, memory allocation, and DLPack.
pub struct QdpEngine {
    device: Arc<CudaDevice>,
    pool: Arc<StagingBufferPool>,
}

impl QdpEngine {
    /// Initialize engine on GPU device
    ///
    /// # Arguments
    /// * `device_id` - CUDA device ID (typically 0)
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| MahoutError::Cuda(format!("Failed to initialize CUDA device {}: {:?}", device_id, e)))?;
        let pool = Arc::new(StagingBufferPool::new(device.clone()));
        Ok(Self {
            device,  // CudaDevice::new already returns Arc<CudaDevice> in cudarc 0.11
            pool,
        })
    }

    /// Encode classical data into quantum state
    ///
    /// Selects encoding strategy, executes on GPU, returns DLPack pointer.
    ///
    /// # Arguments
    /// * `data` - Input data
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// DLPack pointer for zero-copy PyTorch integration
    ///
    /// # Safety
    /// Pointer freed by DLPack deleter, do not free manually.
    pub fn encode(
        &self,
        data: &[f64],
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::Encode");

        let encoder = get_encoder(encoding_method)?;
        let state_vector = encoder.encode(&self.device, &self.pool, data, num_qubits)?;
        let dlpack_ptr = {
            crate::profile_scope!("DLPack::Wrap");
            state_vector.to_dlpack()
        };
        Ok(dlpack_ptr)
    }

    /// Get CUDA device reference for advanced operations
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Encode from chunked Arrow arrays (zero-copy from Parquet)
    ///
    /// # Arguments
    /// * `chunks` - Chunked Arrow Float64Arrays (from read_parquet_to_arrow)
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// DLPack pointer for zero-copy PyTorch integration
    pub fn encode_chunked(
        &self,
        chunks: &[Float64Array],
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeChunked");

        let encoder = get_encoder(encoding_method)?;
        let state_vector = encoder.encode_chunked(&self.device, &self.pool, chunks, num_qubits)?;
        let dlpack_ptr = {
            crate::profile_scope!("Mahout::CreateDLPack");
            state_vector.to_dlpack()
        };
        Ok(dlpack_ptr)
    }

    /// Load data from Parquet file and encode into quantum state
    ///
    /// **ZERO-COPY**: Reads Parquet chunks directly without intermediate Vec allocation.
    ///
    /// # Arguments
    /// * `path` - Path to Parquet file
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    pub fn encode_from_parquet(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromParquet");

        let chunks = crate::io::read_parquet_to_arrow(path)?;
        self.encode_chunked(&chunks, num_qubits, encoding_method)
    }
}

// Re-export key types for convenience
pub use gpu::QuantumEncoder;
