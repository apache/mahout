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
use crate::gpu::topology::DeviceMesh;

/// Runtime distribution modes for distributed state construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionMode {
    Single,
    ShardedCapacity,
    Replicated,
}

/// Placement policy for slicing the logical output state across devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardPolicy {
    /// Evenly partition the logical state into contiguous shards.
    ///
    /// Because the global state length is always `2^n`, equal-width integer
    /// shards are only possible for power-of-two device counts.
    Equal,
    BalancedUneven,
}

/// Planner input describing the logical distributed state request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementRequest {
    pub num_qubits: usize,
    pub mode: DistributionMode,
    pub shard_policy: ShardPolicy,
    pub world_size: usize,
}

impl PlacementRequest {
    /// Create one placement request for one logical distributed state.
    pub fn new(num_qubits: usize, mode: DistributionMode, shard_policy: ShardPolicy) -> Self {
        Self {
            num_qubits,
            mode,
            shard_policy,
            world_size: 1,
        }
    }

    /// Create one placement request with explicit rank-world metadata.
    pub fn new_with_world(
        num_qubits: usize,
        mode: DistributionMode,
        shard_policy: ShardPolicy,
        world_size: usize,
    ) -> Result<Self> {
        if world_size == 0 {
            return Err(MahoutError::InvalidInput(
                "Distributed placement world size must be at least 1".to_string(),
            ));
        }

        Ok(Self {
            num_qubits,
            mode,
            shard_policy,
            world_size,
        })
    }

    /// Compute the global amplitude length implied by `num_qubits`.
    pub fn global_len(&self) -> Result<usize> {
        1usize.checked_shl(self.num_qubits as u32).ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "Global amplitude length overflow for {} qubits",
                self.num_qubits
            ))
        })
    }
}

/// One logical placement decision produced by the planner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardPlacement {
    pub rank_id: usize,
    pub device_id: usize,
    pub shard_id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
}

impl ShardPlacement {
    /// Number of amplitudes assigned to this shard.
    pub fn local_len(&self) -> usize {
        self.end_idx - self.start_idx
    }
}

/// Planner output consumed by distributed encoders and by any later
/// gather/export layer that needs stable shard ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementPlan {
    pub mode: DistributionMode,
    pub global_len: usize,
    pub placements: Vec<ShardPlacement>,
    pub gather_device_id: Option<usize>,
}

impl PlacementPlan {
    /// Number of participating devices in this placement plan.
    pub fn num_devices(&self) -> usize {
        self.placements.len()
    }

    /// Return the common shard length when every shard is evenly sized.
    pub fn shard_len(&self) -> Result<usize> {
        let Some(first) = self.placements.first() else {
            return Err(MahoutError::InvalidInput(
                "Placement plan must contain at least one shard".to_string(),
            ));
        };
        let shard_len = first.local_len();
        if self
            .placements
            .iter()
            .any(|placement| placement.local_len() != shard_len)
        {
            return Err(MahoutError::InvalidInput(
                "Placement plan contains uneven shard lengths".to_string(),
            ));
        }
        Ok(shard_len)
    }

    /// Iterate over placements owned by one rank without allocating.
    pub fn placements_for_rank_iter(&self, rank: usize) -> impl Iterator<Item = &ShardPlacement> {
        self.placements
            .iter()
            .filter(move |placement| placement.rank_id == rank)
    }

    /// Largest shard length owned by one rank.
    pub fn local_max_len(&self, rank: usize) -> usize {
        self.placements_for_rank_iter(rank)
            .map(ShardPlacement::local_len)
            .max()
            .unwrap_or(0)
    }
}

/// Stateless planner for device placement decisions.
#[derive(Clone, Debug, Default)]
pub struct PlacementPlanner;

impl PlacementPlanner {
    /// Build one placement plan from one validated device mesh and request.
    pub fn plan(mesh: &DeviceMesh, request: &PlacementRequest) -> Result<PlacementPlan> {
        Self::validate_inputs(mesh, request)?;

        let global_len = request.global_len()?;
        match request.mode {
            DistributionMode::Single => {
                let device_id = mesh.device_ids.first().copied().ok_or_else(|| {
                    MahoutError::InvalidInput(
                        "Single-device placement requires one device ID".to_string(),
                    )
                })?;
                Ok(PlacementPlan {
                    mode: request.mode,
                    global_len,
                    placements: vec![ShardPlacement {
                        rank_id: 0,
                        device_id,
                        shard_id: 0,
                        start_idx: 0,
                        end_idx: global_len,
                    }],
                    gather_device_id: Some(device_id),
                })
            }
            DistributionMode::ShardedCapacity => {
                let device_ids = Self::select_sharded_device_ids(mesh);
                let shard_lengths =
                    Self::plan_shard_lengths(global_len, device_ids.len(), request.shard_policy)?;
                let placements =
                    Self::build_shard_placements(&device_ids, &shard_lengths, request.world_size);

                Ok(PlacementPlan {
                    mode: request.mode,
                    global_len,
                    placements,
                    gather_device_id: mesh.recommended_gather_device_id(),
                })
            }
            DistributionMode::Replicated => Err(MahoutError::NotImplemented(
                "Replicated placement is not implemented yet".to_string(),
            )),
        }
    }

    /// Build a global placement plan from one rank's local device mesh.
    ///
    /// Placements are emitted shard-major by local device, then rank. For
    /// local devices `[0, 1]` in a two-rank world, the rank/device sequence is
    /// `(0, 0), (1, 0), (0, 1), (1, 1)`.
    pub fn plan_rank_local(mesh: &DeviceMesh, request: &PlacementRequest) -> Result<PlacementPlan> {
        Self::validate_inputs(mesh, request)?;

        if request.mode != DistributionMode::ShardedCapacity {
            return Self::plan(mesh, request);
        }

        let global_len = request.global_len()?;
        let device_ids = Self::select_sharded_device_ids(mesh);
        let total_shards = device_ids
            .len()
            .checked_mul(request.world_size)
            .ok_or_else(|| {
                MahoutError::InvalidInput(format!(
                    "Rank-local shard count overflowed for {} devices across world size {}",
                    device_ids.len(),
                    request.world_size
                ))
            })?;
        let shard_lengths =
            Self::plan_shard_lengths(global_len, total_shards, request.shard_policy)?;
        let placements = Self::build_rank_local_shard_placements(
            &device_ids,
            &shard_lengths,
            request.world_size,
        );

        Ok(PlacementPlan {
            mode: request.mode,
            global_len,
            placements,
            gather_device_id: mesh.recommended_gather_device_id(),
        })
    }

    fn validate_inputs(mesh: &DeviceMesh, request: &PlacementRequest) -> Result<()> {
        if request.world_size == 0 {
            return Err(MahoutError::InvalidInput(
                "Distributed placement world size must be at least 1".to_string(),
            ));
        }

        if mesh.num_devices() == 0 {
            return Err(MahoutError::InvalidInput(
                "Placement planner requires at least one device".to_string(),
            ));
        }

        Ok(())
    }

    fn select_sharded_device_ids(mesh: &DeviceMesh) -> Vec<usize> {
        let recommended = mesh.recommended_placement_device_ids();
        if recommended.is_empty() {
            mesh.device_ids.clone()
        } else {
            recommended
        }
    }

    fn plan_shard_lengths(
        global_len: usize,
        num_devices: usize,
        shard_policy: ShardPolicy,
    ) -> Result<Vec<usize>> {
        if num_devices == 0 {
            return Err(MahoutError::InvalidInput(
                "Shard planning requires at least one device".to_string(),
            ));
        }

        match shard_policy {
            ShardPolicy::Equal => {
                if !num_devices.is_power_of_two() {
                    return Err(MahoutError::InvalidInput(format!(
                        "Equal shard policy requires a power-of-two device count, got {}",
                        num_devices
                    )));
                }
                Ok(vec![global_len / num_devices; num_devices])
            }
            ShardPolicy::BalancedUneven => {
                let base_len = global_len / num_devices;
                let remainder = global_len % num_devices;
                Ok((0..num_devices)
                    .map(|shard_id| base_len + usize::from(shard_id < remainder))
                    .collect())
            }
        }
    }

    fn build_shard_placements(
        device_ids: &[usize],
        shard_lengths: &[usize],
        world_size: usize,
    ) -> Vec<ShardPlacement> {
        let mut start_idx = 0usize;
        device_ids
            .iter()
            .copied()
            .zip(shard_lengths.iter().copied())
            .enumerate()
            .map(|(shard_id, (device_id, local_len))| {
                let end_idx = start_idx + local_len;
                let placement = ShardPlacement {
                    rank_id: shard_id % world_size,
                    device_id,
                    shard_id,
                    start_idx,
                    end_idx,
                };
                start_idx = end_idx;
                placement
            })
            .collect()
    }

    fn build_rank_local_shard_placements(
        device_ids: &[usize],
        shard_lengths: &[usize],
        world_size: usize,
    ) -> Vec<ShardPlacement> {
        let mut start_idx = 0usize;
        device_ids
            .iter()
            .copied()
            .flat_map(|device_id| (0..world_size).map(move |rank_id| (rank_id, device_id)))
            .zip(shard_lengths.iter().copied())
            .enumerate()
            .map(|(shard_id, ((rank_id, device_id), local_len))| {
                let end_idx = start_idx + local_len;
                let placement = ShardPlacement {
                    rank_id,
                    device_id,
                    shard_id,
                    start_idx,
                    end_idx,
                };
                start_idx = end_idx;
                placement
            })
            .collect()
    }
}
