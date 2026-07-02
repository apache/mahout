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
    DistributedStatePlan, DistributionMode, PlacementPlan, PlacementPlanner, PlacementRequest,
};
use crate::gpu::topology::DeviceMesh;
use std::ops::Deref;

/// Amplitude-specific wrapper around the shared distributed state plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedAmplitudePlan {
    pub state: DistributedStatePlan,
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
    /// Validate one distributed amplitude request and derive the shard math used
    /// by later layout and materialization steps.
    pub fn for_request(mesh: &DeviceMesh, request: PlacementRequest) -> Result<Self> {
        let placement = Self::plan_request(mesh, &request, PlacementPlanner::plan)?;
        Self::from_placement(request, placement)
    }

    /// Validate one distributed amplitude request against a rank-local mesh and
    /// derive global shard metadata for all ranks.
    pub fn for_rank_local_request(mesh: &DeviceMesh, request: PlacementRequest) -> Result<Self> {
        let placement = Self::plan_request(mesh, &request, PlacementPlanner::plan_rank_local)?;
        Self::from_placement(request, placement)
    }

    fn plan_request(
        mesh: &DeviceMesh,
        request: &PlacementRequest,
        planner: fn(&DeviceMesh, &PlacementRequest) -> Result<PlacementPlan>,
    ) -> Result<PlacementPlan> {
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

        planner(mesh, request)
    }

    fn from_placement(request: PlacementRequest, placement: PlacementPlan) -> Result<Self> {
        Ok(Self {
            state: DistributedStatePlan::from_placement(request, placement)?,
        })
    }
}

impl AsRef<DistributedStatePlan> for DistributedAmplitudePlan {
    fn as_ref(&self) -> &DistributedStatePlan {
        &self.state
    }
}

impl Deref for DistributedAmplitudePlan {
    type Target = DistributedStatePlan;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}
