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
use crate::gpu::distributed::{
    DistributionMode, PlacementPlan, PlacementPlanner, PlacementRequest, ShardPlacement,
    ShardPolicy,
};
use crate::gpu::encodings::MAX_QUBITS;
use crate::gpu::topology::DeviceMesh;

/// Shared planning math for amplitude-sharded state construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedAmplitudePlan {
    pub request: PlacementRequest,
    pub placement: PlacementPlan,
    pub num_qubits: usize,
    pub global_len: usize,
    pub num_devices: usize,
    pub shard_bits: Option<usize>,
    pub uniform_shard_len: Option<usize>,
}

/// Result of preparing a distributed amplitude encode without yet allocating
/// concrete shard buffers. This fixes the public API surface for later PRs that
/// will populate `state` with real device allocations.
#[derive(Clone)]
pub struct PreparedDistributedAmplitudeEncode {
    pub mesh: DeviceMesh,
    pub plan: DistributedAmplitudePlan,
    pub inv_norm: f64,
    pub layout: super::layout::DistributedStateLayout,
}

impl DistributedAmplitudePlan {
    pub fn for_request(mesh: &DeviceMesh, request: PlacementRequest) -> Result<Self> {
        if request.num_qubits == 0 {
            return Err(MahoutError::InvalidInput(
                "Number of qubits must be at least 1 for distributed amplitude planning"
                    .to_string(),
            ));
        }
        if mesh.num_devices() == 0 {
            return Err(MahoutError::InvalidInput(
                "Distributed amplitude planning requires at least one device".to_string(),
            ));
        }
        if request.mode != DistributionMode::ShardedCapacity {
            return Err(MahoutError::InvalidInput(format!(
                "Distributed amplitude planning currently supports only {:?}, got {:?}",
                DistributionMode::ShardedCapacity,
                request.mode
            )));
        }

        let num_devices = mesh.num_devices();
        let placement = PlacementPlanner::plan(mesh, &request)?;
        Self::validate_local_shard_capacity(request.num_qubits, &placement)?;
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

    pub fn shard_range(&self, shard_id: usize) -> Result<(usize, usize)> {
        let placement = self.placement.placements.get(shard_id).ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "Shard ID {} out of range for {} devices",
                shard_id, self.num_devices
            ))
        })?;
        Ok((placement.start_idx, placement.end_idx))
    }

    fn validate_local_shard_capacity(num_qubits: usize, placement: &PlacementPlan) -> Result<()> {
        let max_local_len = 1usize << MAX_QUBITS;
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

        if required_local_len > max_local_len {
            return Err(MahoutError::InvalidInput(format!(
                "Distributed amplitude request for {} qubits still exceeds per-device capacity: largest shard has {} amplitudes, limit is 2^{} = {}. Add more GPUs or reduce qubits.",
                num_qubits, required_local_len, MAX_QUBITS, max_local_len
            )));
        }

        Ok(())
    }
}
