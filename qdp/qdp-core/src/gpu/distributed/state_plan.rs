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

use crate::error::{MahoutError, Result};
use crate::gpu::distributed::{PlacementPlan, PlacementRequest, ShardPlacement, ShardPolicy};
use crate::gpu::memory::Precision;
use crate::gpu::topology::DeviceMesh;

/// Workload-neutral plan for one logically distributed GPU state.
///
/// Encoder-specific plans should wrap this type instead of duplicating shard
/// placement, rank ownership, and byte-estimation metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedStatePlan {
    request: PlacementRequest,
    placement: PlacementPlan,
    pub num_qubits: usize,
    pub global_len: usize,
    pub num_devices: usize,
    pub shard_bits: Option<usize>,
    pub uniform_shard_len: Option<usize>,
}

impl DistributedStatePlan {
    /// Build shared distributed state metadata from one validated placement.
    pub fn from_placement(request: PlacementRequest, placement: PlacementPlan) -> Result<Self> {
        let num_devices = placement.num_devices();
        Self::validate_local_shard_shape(request.num_qubits, &placement)?;
        let global_len = placement.global_len;
        let num_qubits = request.num_qubits;
        let (shard_bits, uniform_shard_len) = match request.shard_policy {
            ShardPolicy::Equal => {
                debug_assert!(num_devices.is_power_of_two());
                let shard_bits = num_devices.trailing_zeros() as usize;
                if shard_bits > request.num_qubits {
                    return Err(MahoutError::InvalidInput(format!(
                        "Cannot shard {} qubits across {} devices: shard bits {} exceed qubit count",
                        request.num_qubits, num_devices, shard_bits
                    )));
                }
                (Some(shard_bits), Some(placement.shard_len()?))
            }
            ShardPolicy::BalancedUneven => (None, None),
        };

        Ok(Self {
            request,
            placement,
            num_qubits,
            global_len,
            num_devices,
            shard_bits,
            uniform_shard_len,
        })
    }

    /// Original placement request used to build this distributed state plan.
    pub fn request(&self) -> &PlacementRequest {
        &self.request
    }

    /// Raw planner output backing this shared state plan.
    pub fn placement(&self) -> &PlacementPlan {
        &self.placement
    }

    /// Logical half-open state range covered by one shard ID.
    pub fn shard_range(&self, shard_id: usize) -> Result<(usize, usize)> {
        let placement = self.placement.placements.get(shard_id).ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "Shard ID {} out of range for {} devices",
                shard_id, self.num_devices
            ))
        })?;
        Ok((placement.start_idx, placement.end_idx))
    }

    /// Largest local shard length across the current placement.
    pub fn max_local_len(&self) -> usize {
        self.placement
            .placements
            .iter()
            .map(ShardPlacement::local_len)
            .max()
            .unwrap_or(0)
    }

    /// Iterate over placements owned by one rank without allocating.
    pub fn placements_for_rank_iter(&self, rank: usize) -> impl Iterator<Item = &ShardPlacement> {
        self.placement.placements_for_rank_iter(rank)
    }

    /// Number of shards owned by one rank.
    pub fn num_local_shards(&self, rank: usize) -> usize {
        self.placements_for_rank_iter(rank).count()
    }

    /// Logical half-open state ranges owned by one rank.
    pub fn rank_shard_ranges(&self, rank: usize) -> Vec<(usize, usize)> {
        self.placements_for_rank_iter(rank)
            .map(|placement| (placement.start_idx, placement.end_idx))
            .collect()
    }

    /// Largest shard length owned by one rank.
    pub fn max_local_len_for_rank(&self, rank: usize) -> usize {
        self.placements_for_rank_iter(rank)
            .map(ShardPlacement::local_len)
            .max()
            .unwrap_or(0)
    }

    /// Estimated bytes required by the largest local shard at one target precision.
    pub fn estimated_max_shard_bytes(&self, precision: Precision) -> Result<usize> {
        estimated_state_bytes(self.max_local_len(), precision)
    }

    /// Validate that this plan can be materialized by the provided local mesh.
    pub fn validate_against_mesh(&self, mesh: &DeviceMesh) -> Result<()> {
        if self.placement.placements.len() != self.num_devices {
            return Err(MahoutError::InvalidInput(format!(
                "Placement plan mismatch: {} placements for {} planned shards",
                self.placement.placements.len(),
                self.num_devices
            )));
        }
        if mesh.devices.len() != mesh.device_ids.len() {
            return Err(MahoutError::InvalidInput(format!(
                "Device mesh / device handles mismatch: {} handles for {} device IDs",
                mesh.devices.len(),
                mesh.device_ids.len()
            )));
        }

        Ok(())
    }

    fn validate_local_shard_shape(num_qubits: usize, placement: &PlacementPlan) -> Result<()> {
        let required_local_len = placement
            .placements
            .iter()
            .map(ShardPlacement::local_len)
            .max()
            .ok_or_else(|| {
                MahoutError::InvalidInput(
                    "Placement plan must contain at least one shard".to_string(),
                )
            })?;

        if required_local_len == 0 {
            return Err(MahoutError::InvalidInput(format!(
                "Distributed state request for {} qubits produced an empty local shard",
                num_qubits
            )));
        }

        let _ = estimated_state_bytes(required_local_len, Precision::Float32)?;
        let _ = estimated_state_bytes(required_local_len, Precision::Float64)?;

        Ok(())
    }
}

fn estimated_state_bytes(local_len: usize, precision: Precision) -> Result<usize> {
    let bytes_per_state_entry = match precision {
        Precision::Float32 => 8usize,
        Precision::Float64 => 16usize,
    };

    local_len.checked_mul(bytes_per_state_entry).ok_or_else(|| {
        MahoutError::InvalidInput(format!(
            "Distributed state shard byte estimate overflowed for local_len={} and precision={:?}",
            local_len, precision
        ))
    })
}
