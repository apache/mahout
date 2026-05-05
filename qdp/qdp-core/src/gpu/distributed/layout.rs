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

use crate::error::{MahoutError, Result};
use crate::gpu::memory::Precision;
use crate::gpu::topology::{DeviceMesh, GpuTopology};

use super::DistributedAmplitudePlan;
use super::shared;

/// One shard of a logically distributed state vector.
#[derive(Clone)]
pub struct StateShardLayout {
    pub device: Arc<CudaDevice>,
    pub device_id: usize,
    pub shard_id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub local_len: usize,
}

/// Metadata describing how one distributed state is mapped onto one execution
/// context.
#[derive(Clone)]
pub struct DistributedStateLayout {
    pub num_qubits: usize,
    pub precision: Precision,
    pub global_len: usize,
    pub shard_bits: Option<usize>,
    pub topology: GpuTopology,
    pub shards: Vec<StateShardLayout>,
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

    /// Build one distributed state layout from one execution mesh and one
    /// distributed amplitude plan.
    pub fn new(
        mesh: &DeviceMesh,
        plan: &DistributedAmplitudePlan,
        precision: Precision,
    ) -> Result<Self> {
        if mesh.num_devices() != plan.num_devices {
            return Err(MahoutError::InvalidInput(format!(
                "Device mesh / amplitude plan mismatch: {} devices vs {} planned shards",
                mesh.num_devices(),
                plan.num_devices
            )));
        }
        if mesh.devices.len() != plan.num_devices {
            return Err(MahoutError::InvalidInput(format!(
                "Device mesh / device handles mismatch: {} handles for {} planned shards",
                mesh.devices.len(),
                plan.num_devices
            )));
        }
        if plan.placement.placements.len() != plan.num_devices {
            return Err(MahoutError::InvalidInput(format!(
                "Placement plan mismatch: {} placements for {} planned shards",
                plan.placement.placements.len(),
                plan.num_devices
            )));
        }

        let mut shards = Vec::with_capacity(mesh.num_devices());
        for placement in &plan.placement.placements {
            let (start_idx, end_idx) = (placement.start_idx, placement.end_idx);
            shards.push(StateShardLayout {
                device: mesh.device_for_id(placement.device_id)?,
                device_id: placement.device_id,
                shard_id: placement.shard_id,
                start_idx,
                end_idx,
                local_len: placement.local_len(),
            });
        }

        Ok(Self {
            num_qubits: plan.num_qubits,
            precision,
            global_len: plan.global_len,
            shard_bits: plan.shard_bits,
            topology: mesh.topology.clone(),
            shards,
        })
    }
}
