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

/// Runtime distribution modes for future multi-GPU state construction.
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
}

impl PlacementRequest {
    pub fn new(num_qubits: usize, mode: DistributionMode, shard_policy: ShardPolicy) -> Self {
        Self {
            num_qubits,
            mode,
            shard_policy,
        }
    }

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
    pub device_id: usize,
    pub shard_id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
}

impl ShardPlacement {
    pub fn local_len(&self) -> usize {
        self.end_idx - self.start_idx
    }
}

/// Planner output consumed by distributed encoders and future gather/export APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementPlan {
    pub mode: DistributionMode,
    pub global_len: usize,
    pub placements: Vec<ShardPlacement>,
    pub gather_device_id: Option<usize>,
}

impl PlacementPlan {
    pub fn num_devices(&self) -> usize {
        self.placements.len()
    }

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
}

/// Stateless planner for device placement decisions.
#[derive(Clone, Debug, Default)]
pub struct PlacementPlanner;

impl PlacementPlanner {
    pub fn plan(mesh: &DeviceMesh, request: &PlacementRequest) -> Result<PlacementPlan> {
        if mesh.num_devices() == 0 {
            return Err(MahoutError::InvalidInput(
                "Placement planner requires at least one device".to_string(),
            ));
        }

        let global_len = request.global_len()?;
        match request.mode {
            DistributionMode::Single => {
                let device_ids = Self::select_device_ids(mesh, request)?;
                let device_id = *device_ids.first().ok_or_else(|| {
                    MahoutError::InvalidInput(
                        "Single-device placement requires one device ID".to_string(),
                    )
                })?;
                Ok(PlacementPlan {
                    mode: request.mode,
                    global_len,
                    placements: vec![ShardPlacement {
                        device_id,
                        shard_id: 0,
                        start_idx: 0,
                        end_idx: global_len,
                    }],
                    gather_device_id: Some(device_id),
                })
            }
            DistributionMode::ShardedCapacity => {
                let device_ids = Self::select_device_ids(mesh, request)?;
                let shard_lengths =
                    Self::plan_shard_lengths(global_len, device_ids.len(), request.shard_policy)?;
                let placements = Self::build_shard_placements(&device_ids, &shard_lengths);

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

    fn select_device_ids(mesh: &DeviceMesh, request: &PlacementRequest) -> Result<Vec<usize>> {
        match request.mode {
            DistributionMode::Single => mesh
                .device_ids
                .first()
                .copied()
                .map(|device_id| vec![device_id])
                .ok_or_else(|| {
                    MahoutError::InvalidInput(
                        "Single-device placement requires one device ID".to_string(),
                    )
                }),
            DistributionMode::ShardedCapacity => {
                let recommended = mesh.recommended_placement_device_ids();
                if recommended.is_empty() {
                    Ok(mesh.device_ids.clone())
                } else {
                    Ok(recommended)
                }
            }
            DistributionMode::Replicated => Err(MahoutError::NotImplemented(
                "Replicated placement is not implemented yet".to_string(),
            )),
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
