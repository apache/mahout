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

use crate::error::Result;
use crate::gpu::communicator::CollectiveCommunicator;
use crate::gpu::topology::DeviceMesh;

/// Bundles the device mesh with the collective implementation that coordinates
/// those devices.
///
/// The current branch uses one process with one mesh covering all participating
/// devices. A future MPI implementation can construct one context per rank with
/// a rank-local mesh and an MPI-backed collective implementation.
pub struct DistributedExecutionContext<'a> {
    mesh: DeviceMesh,
    collectives: &'a dyn CollectiveCommunicator,
}

impl<'a> DistributedExecutionContext<'a> {
    pub fn new(mesh: DeviceMesh, collectives: &'a dyn CollectiveCommunicator) -> Self {
        Self { mesh, collectives }
    }

    pub fn single_process(
        device_ids: Vec<usize>,
        collectives: &'a dyn CollectiveCommunicator,
    ) -> Result<Self> {
        Ok(Self::new(DeviceMesh::new(device_ids)?, collectives))
    }

    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    pub(crate) fn collectives(&self) -> &dyn CollectiveCommunicator {
        self.collectives
    }
}
