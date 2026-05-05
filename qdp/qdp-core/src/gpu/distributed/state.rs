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

use std::sync::Arc;

use cudarc::driver::CudaDevice;
#[cfg(target_os = "linux")]
use qdp_kernels::{CuComplex, CuDoubleComplex};

use crate::error::{MahoutError, Result};
use crate::gpu::memory::{BufferStorage, Precision};

use super::DistributedStateLayout;
use super::shared;

/// One materialized shard of a distributed state vector.
#[derive(Clone)]
pub struct StateShard {
    pub device: Arc<CudaDevice>,
    pub device_id: usize,
    pub shard_id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub local_len: usize,
    pub buffer: Arc<BufferStorage>,
}

/// Materialized multi-GPU state vector with one live buffer per shard.
#[derive(Clone)]
pub struct DistributedStateVector {
    pub num_qubits: usize,
    pub precision: Precision,
    pub global_len: usize,
    pub shard_bits: Option<usize>,
    pub topology: crate::gpu::topology::GpuTopology,
    pub shards: Vec<StateShard>,
}

impl DistributedStateVector {
    /// Device ID preferred for future gather-style readback operations.
    pub fn recommended_gather_device_id(&self) -> Option<usize> {
        shared::policy_device_ids(
            &self.topology,
            self.shards.iter().map(|shard| shard.device_id),
        )
        .0
    }

    /// Preferred device ordering derived from the current topology metadata.
    pub fn recommended_placement_device_ids(&self) -> Vec<usize> {
        shared::policy_device_ids(
            &self.topology,
            self.shards.iter().map(|shard| shard.device_id),
        )
        .1
    }

    /// Number of materialized shards in this distributed state.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    #[cfg(target_os = "linux")]
    /// Copy one float64 shard back to host memory for validation or inspection.
    pub fn copy_shard_to_host_f64(&self, shard_id: usize) -> Result<Vec<CuDoubleComplex>> {
        let shard = self.shards.get(shard_id).ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "Shard ID {} out of range for {} shards",
                shard_id,
                self.shards.len()
            ))
        })?;
        match shard.buffer.as_ref() {
            BufferStorage::F64(buf) => shard.device.dtoh_sync_copy(&buf.slice).map_err(|e| {
                MahoutError::Cuda(format!(
                    "Failed to copy distributed float64 shard {} to host: {:?}",
                    shard_id, e
                ))
            }),
            BufferStorage::F32(_) => Err(MahoutError::InvalidInput(format!(
                "Shard {} stores float32 data, not float64",
                shard_id
            ))),
        }
    }

    #[cfg(target_os = "linux")]
    /// Copy one float32 shard back to host memory for validation or inspection.
    pub fn copy_shard_to_host_f32(&self, shard_id: usize) -> Result<Vec<CuComplex>> {
        let shard = self.shards.get(shard_id).ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "Shard ID {} out of range for {} shards",
                shard_id,
                self.shards.len()
            ))
        })?;
        match shard.buffer.as_ref() {
            BufferStorage::F32(buf) => shard.device.dtoh_sync_copy(&buf.slice).map_err(|e| {
                MahoutError::Cuda(format!(
                    "Failed to copy distributed float32 shard {} to host: {:?}",
                    shard_id, e
                ))
            }),
            BufferStorage::F64(_) => Err(MahoutError::InvalidInput(format!(
                "Shard {} stores float64 data, not float32",
                shard_id
            ))),
        }
    }

    /// Construct a distributed state vector with concrete shard buffers already allocated.
    pub fn new_with_buffers(
        layout: DistributedStateLayout,
        buffers: Vec<Arc<BufferStorage>>,
    ) -> Result<Self> {
        if buffers.len() != layout.shards.len() {
            return Err(MahoutError::InvalidInput(format!(
                "Distributed state buffer mismatch: {} buffers for {} shards",
                buffers.len(),
                layout.shards.len()
            )));
        }

        let mut shards = Vec::with_capacity(layout.shards.len());
        for (shard_layout, buffer) in layout.shards.into_iter().zip(buffers.into_iter()) {
            if buffer.precision() != layout.precision {
                return Err(MahoutError::InvalidInput(format!(
                    "Distributed shard precision mismatch on shard {}: expected {:?}, got {:?}",
                    shard_layout.shard_id,
                    layout.precision,
                    buffer.precision()
                )));
            }
            shards.push(StateShard {
                device: shard_layout.device,
                device_id: shard_layout.device_id,
                shard_id: shard_layout.shard_id,
                start_idx: shard_layout.start_idx,
                end_idx: shard_layout.end_idx,
                local_len: shard_layout.local_len,
                buffer,
            });
        }

        Ok(Self {
            num_qubits: layout.num_qubits,
            precision: layout.precision,
            global_len: layout.global_len,
            shard_bits: layout.shard_bits,
            topology: layout.topology,
            shards,
        })
    }
}
