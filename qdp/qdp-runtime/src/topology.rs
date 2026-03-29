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

use qdp_core::{MahoutError, Result};

use crate::worker::WorkerRegistration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostPlatform {
    Linux,
    Wsl,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterconnectKind {
    Nvlink,
    Pcie,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerLink {
    pub peer_node_id: String,
    pub peer_device_id: usize,
    pub kind: InterconnectKind,
    pub bandwidth_rank: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceCapabilities {
    pub node_id: String,
    pub device_id: usize,
    pub device_name: String,
    pub total_memory_bytes: u64,
    pub free_memory_bytes: u64,
    pub max_safe_allocation_bytes: u64,
    pub measured_encode_samples_per_sec: Option<f64>,
    pub host_platform: HostPlatform,
    pub stability_factor: f64,
    pub peer_links: Vec<PeerLink>,
}

impl DeviceCapabilities {
    pub(crate) fn placement_weight(&self) -> Result<f64> {
        if self.total_memory_bytes == 0 {
            return Err(MahoutError::InvalidInput(format!(
                "Device {} on node {} has zero total memory",
                self.device_id, self.node_id
            )));
        }
        if !(0.0 < self.stability_factor && self.stability_factor <= 1.0) {
            return Err(MahoutError::InvalidInput(format!(
                "Device {} on node {} has invalid stability_factor {}",
                self.device_id, self.node_id, self.stability_factor
            )));
        }

        let memory_factor = self.max_safe_allocation_bytes.min(self.free_memory_bytes) as f64
            / self.total_memory_bytes as f64;
        let throughput_factor = self.measured_encode_samples_per_sec.unwrap_or(1.0).max(1.0);
        let host_penalty = match self.host_platform {
            HostPlatform::Linux => 1.0,
            HostPlatform::Wsl => 0.8,
            HostPlatform::Other => 0.7,
        };

        Ok(memory_factor.max(0.05) * throughput_factor * host_penalty * self.stability_factor)
    }

    pub(crate) fn nvlink_peer_count(&self) -> usize {
        self.peer_links
            .iter()
            .filter(|link| link.kind == InterconnectKind::Nvlink)
            .count()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterInventory {
    pub workers: Vec<WorkerRegistration>,
    pub devices: Vec<DeviceCapabilities>,
}

impl ClusterInventory {
    pub fn from_workers(workers: impl IntoIterator<Item = WorkerRegistration>) -> Self {
        let workers = workers.into_iter().collect::<Vec<_>>();
        let devices = workers
            .iter()
            .flat_map(|worker| worker.devices.iter().cloned())
            .collect();
        Self { workers, devices }
    }

    pub fn devices(&self) -> &[DeviceCapabilities] {
        &self.devices
    }

    pub fn worker_for_device(&self, node_id: &str, device_id: usize) -> Option<&WorkerRegistration> {
        self.workers.iter().find(|worker| {
            worker.node_id == node_id
                && worker
                    .devices
                    .iter()
                    .any(|device| device.node_id == node_id && device.device_id == device_id)
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceTopology {
    pub devices: Vec<DeviceCapabilities>,
}

impl DeviceTopology {
    pub fn from_inventory(inventory: &ClusterInventory) -> Self {
        Self {
            devices: inventory.devices.clone(),
        }
    }

    pub fn nvlink_peers(&self, node_id: &str, device_id: usize) -> Vec<&PeerLink> {
        self.devices
            .iter()
            .find(|device| device.node_id == node_id && device.device_id == device_id)
            .map(|device| {
                device
                    .peer_links
                    .iter()
                    .filter(|link| link.kind == InterconnectKind::Nvlink)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn same_nvlink_island(
        &self,
        left_node_id: &str,
        left_device_id: usize,
        right_node_id: &str,
        right_device_id: usize,
    ) -> bool {
        self.nvlink_peers(left_node_id, left_device_id)
            .into_iter()
            .any(|peer| peer.peer_node_id == right_node_id && peer.peer_device_id == right_device_id)
    }
}
