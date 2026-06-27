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

use std::fmt;
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::error::{MahoutError, Result};
use crate::gpu::memory::Precision;
use crate::gpu::topology::{DeviceMesh, GpuTopology};

use super::DistributedStatePlan;
use super::shared;

/// One shard of a logically distributed state vector.
#[derive(Clone)]
pub struct StateShardLayout {
    pub rank_id: usize,
    pub device: Arc<CudaDevice>,
    pub device_id: usize,
    pub shard_id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub local_len: usize,
}

impl fmt::Debug for StateShardLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateShardLayout")
            .field("rank_id", &self.rank_id)
            .field("device_id", &self.device_id)
            .field("shard_id", &self.shard_id)
            .field("start_idx", &self.start_idx)
            .field("end_idx", &self.end_idx)
            .field("local_len", &self.local_len)
            .finish()
    }
}

/// Metadata describing how one distributed state is mapped onto one execution
/// context.
#[derive(Clone)]
pub struct DistributedStateLayout {
    pub rank_id: usize,
    pub world_size: usize,
    pub num_qubits: usize,
    pub precision: Precision,
    pub global_len: usize,
    pub shard_bits: Option<usize>,
    pub topology: GpuTopology,
    shards: Vec<StateShardLayout>,
}

impl fmt::Debug for DistributedStateLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DistributedStateLayout")
            .field("rank_id", &self.rank_id)
            .field("world_size", &self.world_size)
            .field("num_qubits", &self.num_qubits)
            .field("precision", &self.precision)
            .field("global_len", &self.global_len)
            .field("shard_bits", &self.shard_bits)
            .field("shards", &self.shards)
            .finish()
    }
}

impl DistributedStateLayout {
    #[cfg(target_os = "linux")]
    /// Device ID preferred for future gather-style readback operations.
    pub fn recommended_gather_device_id(&self) -> Option<usize> {
        shared::policy_device_ids(
            &self.topology,
            self.shards.iter().map(|shard| shard.device_id),
        )
        .0
    }

    #[cfg(target_os = "linux")]
    /// Preferred device ordering derived from the current topology metadata.
    pub fn recommended_placement_device_ids(&self) -> Vec<usize> {
        shared::policy_device_ids(
            &self.topology,
            self.shards.iter().map(|shard| shard.device_id),
        )
        .1
    }

    /// Number of logical shards described by this layout.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Borrow all rank-local shard layout records.
    pub fn shards(&self) -> &[StateShardLayout] {
        &self.shards
    }

    /// Iterate over rank-local shard layout records.
    pub fn iter_shards(&self) -> impl Iterator<Item = &StateShardLayout> {
        self.shards.iter()
    }

    pub(crate) fn into_shards(self) -> Vec<StateShardLayout> {
        self.shards
    }

    /// Build one distributed state layout from one execution mesh and one
    /// distributed state plan.
    pub fn new(
        mesh: &DeviceMesh,
        plan: &(impl AsRef<DistributedStatePlan> + ?Sized),
        precision: Precision,
    ) -> Result<Self> {
        Self::new_for_rank(mesh, plan, precision, 0)
    }

    /// Build one rank-local distributed state layout from placements owned by
    /// `rank_id`.
    pub fn new_for_rank(
        mesh: &DeviceMesh,
        plan: &(impl AsRef<DistributedStatePlan> + ?Sized),
        precision: Precision,
        rank_id: usize,
    ) -> Result<Self> {
        let plan = plan.as_ref();
        if rank_id >= plan.request().world_size {
            return Err(MahoutError::InvalidInput(format!(
                "rank {} is out of range for world size {}",
                rank_id,
                plan.request().world_size
            )));
        }

        plan.validate_against_mesh(mesh)?;

        let mut shards = Vec::with_capacity(plan.num_local_shards(rank_id));
        for placement in plan.placements_for_rank_iter(rank_id) {
            let (start_idx, end_idx) = (placement.start_idx, placement.end_idx);
            shards.push(StateShardLayout {
                rank_id: placement.rank_id,
                device: mesh.device_for_id(placement.device_id)?,
                device_id: placement.device_id,
                shard_id: placement.shard_id,
                start_idx,
                end_idx,
                local_len: placement.local_len(),
            });
        }

        Ok(Self {
            rank_id,
            world_size: plan.request().world_size,
            num_qubits: plan.num_qubits,
            precision,
            global_len: plan.global_len,
            shard_bits: plan.shard_bits,
            topology: mesh.topology.clone(),
            shards,
        })
    }
}
