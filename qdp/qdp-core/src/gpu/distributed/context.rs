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

use crate::error::{MahoutError, Result};
use crate::gpu::communicator::{CollectiveCommunicator, DeviceCollectiveCommunicator};
use crate::gpu::topology::DeviceMesh;

/// Bundles the device mesh with the collective implementation that coordinates
/// those devices.
///
/// The current branch uses one process with one mesh covering all participating
/// devices. A future MPI implementation can construct one context per rank with
/// a rank-local mesh and an MPI-backed collective implementation.
pub struct DistributedExecutionContext<'a> {
    rank: usize,
    world_size: usize,
    mesh: DeviceMesh,
    collectives: &'a dyn CollectiveCommunicator,
    device_collectives: Option<&'a dyn DeviceCollectiveCommunicator>,
}

impl fmt::Debug for DistributedExecutionContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DistributedExecutionContext")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .field("device_ids", &self.mesh.device_ids)
            .field("has_device_collectives", &self.device_collectives.is_some())
            .finish()
    }
}

impl<'a> DistributedExecutionContext<'a> {
    /// Build one distributed execution context from a validated device mesh and
    /// one collective implementation.
    pub fn new(mesh: DeviceMesh, collectives: &'a dyn CollectiveCommunicator) -> Self {
        Self {
            rank: collectives.rank(),
            world_size: collectives.world_size(),
            mesh,
            collectives,
            device_collectives: None,
        }
    }

    /// Build the current single-process execution context from one caller-owned
    /// device list.
    pub fn single_process(
        device_ids: Vec<usize>,
        collectives: &'a dyn CollectiveCommunicator,
    ) -> Result<Self> {
        Ok(Self {
            rank: 0,
            world_size: 1,
            mesh: DeviceMesh::new(device_ids)?,
            collectives,
            device_collectives: None,
        })
    }

    /// Build one rank-local execution context from the CUDA devices owned by
    /// this rank.
    pub fn rank_local(
        rank: usize,
        world_size: usize,
        device_ids: Vec<usize>,
        collectives: &'a dyn CollectiveCommunicator,
    ) -> Result<Self> {
        Self::validate_rank_world(rank, world_size)?;
        Ok(Self {
            rank,
            world_size,
            mesh: DeviceMesh::new(device_ids)?,
            collectives,
            device_collectives: None,
        })
    }

    /// Rank represented by this execution context.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Total number of ranks participating in this distributed execution.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Access the device mesh that owns the distributed state shards.
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Access scalar collectives used for rank-level reductions.
    pub fn collectives(&self) -> &dyn CollectiveCommunicator {
        self.collectives
    }

    /// Access optional CUDA-aware device collectives.
    pub fn device_collectives(&self) -> Option<&dyn DeviceCollectiveCommunicator> {
        self.device_collectives
    }

    fn validate_rank_world(rank: usize, world_size: usize) -> Result<()> {
        if world_size == 0 || rank >= world_size {
            return Err(MahoutError::InvalidInput(format!(
                "Distributed execution rank {} must be within world size {}",
                rank, world_size
            )));
        }

        Ok(())
    }
}
