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
#[cfg(target_os = "linux")]
use crate::{
    cuda_error_to_string,
    gpu::cuda_ffi::{CUDA_SUCCESS, cudaDeviceCanAccessPeer},
};

/// Coarse-grained GPU interconnect classification used by placement policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkKind {
    SameDevice,
    Pix,
    Node,
    Sys,
    Unknown,
}

impl LinkKind {
    fn score(self) -> usize {
        match self {
            LinkKind::SameDevice => 4,
            LinkKind::Pix => 3,
            LinkKind::Node => 2,
            LinkKind::Sys => 1,
            LinkKind::Unknown => 0,
        }
    }
}

/// Runtime topology metadata for a device mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuTopology {
    pub peer_access: Vec<Vec<bool>>,
    pub links: Vec<Vec<LinkKind>>,
}

impl GpuTopology {
    /// Create a placeholder topology where only self-access is guaranteed.
    pub fn placeholder(num_devices: usize) -> Self {
        let mut peer_access = vec![vec![false; num_devices]; num_devices];
        let mut links = vec![vec![LinkKind::Unknown; num_devices]; num_devices];

        for idx in 0..num_devices {
            peer_access[idx][idx] = true;
            links[idx][idx] = LinkKind::SameDevice;
        }

        Self { peer_access, links }
    }

    #[cfg(target_os = "linux")]
    /// Probe peer-access reachability for one caller-provided device list.
    pub fn probe(device_ids: &[usize]) -> Result<Self> {
        let num_devices = device_ids.len();
        let mut topology = Self::placeholder(num_devices);

        for (src_idx, &src_device_id) in device_ids.iter().enumerate() {
            for (dst_idx, &dst_device_id) in device_ids.iter().enumerate() {
                if src_idx == dst_idx {
                    continue;
                }

                let mut can_access_peer = 0i32;
                let ret = unsafe {
                    cudaDeviceCanAccessPeer(
                        &mut can_access_peer as *mut i32,
                        src_device_id as i32,
                        dst_device_id as i32,
                    )
                };

                if ret != CUDA_SUCCESS {
                    return Err(MahoutError::Cuda(format!(
                        "cudaDeviceCanAccessPeer(cuda:{}, cuda:{}) failed: {} ({})",
                        src_device_id,
                        dst_device_id,
                        ret,
                        cuda_error_to_string(ret)
                    )));
                }

                topology.peer_access[src_idx][dst_idx] = can_access_peer != 0;
            }
        }

        Ok(topology)
    }

    /// Choose the most connected device index as the preferred gather root.
    pub fn preferred_gather_index(&self) -> Option<usize> {
        if self.peer_access.is_empty() {
            return None;
        }

        let mut best_idx = 0usize;
        let mut best_score = 0usize;
        for idx in 0..self.peer_access.len() {
            let mut score = 0usize;
            for peer_idx in 0..self.peer_access.len() {
                if self.peer_access[idx][peer_idx] {
                    score += 10 + self.links[idx][peer_idx].score();
                }
                if self.peer_access[peer_idx][idx] {
                    score += 10 + self.links[peer_idx][idx].score();
                }
            }

            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        Some(best_idx)
    }

    /// Rank device indices by how favorable they are for distributed placement.
    pub fn recommended_placement_order(&self) -> Vec<usize> {
        let Some(root_idx) = self.preferred_gather_index() else {
            return Vec::new();
        };

        let mut indices: Vec<usize> = (0..self.peer_access.len()).collect();
        indices.sort_by_key(|&idx| {
            let mut score = 0usize;
            if self.peer_access[root_idx][idx] {
                score += 10 + self.links[root_idx][idx].score();
            }
            if self.peer_access[idx][root_idx] {
                score += 10 + self.links[idx][root_idx].score();
            }
            (usize::MAX - score, idx)
        });
        indices
    }
}

/// A validated collection of CUDA devices that back one distributed execution context.
#[derive(Clone)]
pub struct DeviceMesh {
    pub device_ids: Vec<usize>,
    pub devices: Vec<Arc<CudaDevice>>,
    pub topology: GpuTopology,
}

impl DeviceMesh {
    /// Build a mesh from CUDA device IDs.
    ///
    /// On Linux this probes peer-access reachability so placement policies can
    /// prefer the most connected devices without changing the caller-provided
    /// device set.
    pub fn new(device_ids: Vec<usize>) -> Result<Self> {
        Self::validate_device_ids(&device_ids)?;

        let mut devices = Vec::with_capacity(device_ids.len());
        for &device_id in &device_ids {
            let device = CudaDevice::new(device_id).map_err(|e| {
                MahoutError::Cuda(format!(
                    "Failed to initialize CUDA device {} for device mesh: {:?}",
                    device_id, e
                ))
            })?;
            devices.push(device);
        }

        #[cfg(target_os = "linux")]
        let topology = GpuTopology::probe(&device_ids)?;
        #[cfg(not(target_os = "linux"))]
        let topology = GpuTopology::placeholder(device_ids.len());

        Ok(Self {
            device_ids,
            devices,
            topology,
        })
    }

    /// Build a mesh from explicit parts. Intended for tests and injected
    /// topology metadata.
    pub fn from_parts(
        device_ids: Vec<usize>,
        devices: Vec<Arc<CudaDevice>>,
        topology: GpuTopology,
    ) -> Result<Self> {
        Self::validate_device_ids(&device_ids)?;

        if device_ids.len() != devices.len() {
            return Err(MahoutError::InvalidInput(format!(
                "Device mesh mismatch: {} device IDs but {} device handles",
                device_ids.len(),
                devices.len()
            )));
        }
        for (&device_id, device) in device_ids.iter().zip(devices.iter()) {
            if device.ordinal() != device_id {
                return Err(MahoutError::InvalidInput(format!(
                    "Device mesh ordinal mismatch: declared cuda:{} but handle targets cuda:{}",
                    device_id,
                    device.ordinal()
                )));
            }
        }

        let num_devices = device_ids.len();
        if topology.peer_access.len() != num_devices || topology.links.len() != num_devices {
            return Err(MahoutError::InvalidInput(format!(
                "Topology dimension mismatch for {} devices",
                num_devices
            )));
        }
        if topology
            .peer_access
            .iter()
            .any(|row| row.len() != num_devices)
            || topology.links.iter().any(|row| row.len() != num_devices)
        {
            return Err(MahoutError::InvalidInput(format!(
                "Topology row width mismatch for {} devices",
                num_devices
            )));
        }

        Ok(Self {
            device_ids,
            devices,
            topology,
        })
    }

    /// Number of devices represented by this mesh.
    pub fn num_devices(&self) -> usize {
        self.device_ids.len()
    }

    /// Enforce a power-of-two device count for policies that require equal shards.
    pub fn validate_power_of_two(&self) -> Result<()> {
        let num_devices = self.num_devices();
        if !num_devices.is_power_of_two() {
            return Err(MahoutError::InvalidInput(format!(
                "Distributed QDP currently requires a power-of-two device count, got {}",
                num_devices
            )));
        }
        Ok(())
    }

    /// Recommended gather device ID derived from the current topology metadata.
    pub fn recommended_gather_device_id(&self) -> Option<usize> {
        self.topology
            .preferred_gather_index()
            .map(|idx| self.device_ids[idx])
    }

    /// Preferred device IDs for placement in topology-aware order.
    pub fn recommended_placement_device_ids(&self) -> Vec<usize> {
        self.topology
            .recommended_placement_order()
            .into_iter()
            .map(|idx| self.device_ids[idx])
            .collect()
    }

    /// Resolve one CUDA device handle by declared device ID.
    pub fn device_for_id(&self, device_id: usize) -> Result<Arc<CudaDevice>> {
        let index = self
            .device_ids
            .iter()
            .position(|&candidate| candidate == device_id)
            .ok_or_else(|| {
                MahoutError::InvalidInput(format!(
                    "Device mesh does not contain cuda:{}",
                    device_id
                ))
            })?;
        Ok(Arc::clone(&self.devices[index]))
    }

    fn validate_device_ids(device_ids: &[usize]) -> Result<()> {
        if device_ids.is_empty() {
            return Err(MahoutError::InvalidInput(
                "Device mesh requires at least one device ID".to_string(),
            ));
        }

        let mut sorted = device_ids.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != device_ids.len() {
            return Err(MahoutError::InvalidInput(
                "Device mesh contains duplicate device IDs".to_string(),
            ));
        }

        Ok(())
    }
}
