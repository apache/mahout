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

use crate::model::{
    ConsumptionMode, DType, DistributedStateHandle, PartitionAssignment, PartitionLayout,
    PlacementPolicy, StatePartitionRef,
};
use crate::runtime_profile_scope;
use crate::topology::DeviceCapabilities;

pub struct DistributedRuntimePlanner;

impl DistributedRuntimePlanner {
    pub fn plan_partitions(
        partition_count: usize,
        devices: &[DeviceCapabilities],
        policy: PlacementPolicy,
    ) -> Result<Vec<PartitionAssignment>> {
        runtime_profile_scope!("Runtime::PlanPartitions");

        if devices.is_empty() {
            return Err(MahoutError::InvalidInput(
                "at least one device is required for partition planning".to_string(),
            ));
        }

        let assignments = match policy {
            PlacementPolicy::RoundRobin => Self::plan_round_robin(partition_count, devices),
            PlacementPolicy::Weighted => Self::plan_weighted(partition_count, devices)?,
            PlacementPolicy::TopologyAware => Self::plan_topology_aware(partition_count, devices)?,
        };
        Ok(assignments)
    }

    pub fn build_state_handle(
        state_id: impl Into<String>,
        global_qubits: u32,
        dtype: DType,
        consumption_mode: ConsumptionMode,
        devices: &[DeviceCapabilities],
        policy: PlacementPolicy,
    ) -> Result<DistributedStateHandle> {
        runtime_profile_scope!("Runtime::BuildDistributedState");

        let state_id = state_id.into();
        let partition_count = devices.len().next_power_of_two();
        let layout = PartitionLayout::new(global_qubits, partition_count)?;
        let assignments = Self::plan_partitions(layout.partition_count, devices, policy)?;

        let partitions = assignments
            .into_iter()
            .map(|assignment| StatePartitionRef {
                storage_handle: format!("state:{}:partition:{}", state_id, assignment.partition_id),
                state_id: state_id.clone(),
                partition_id: assignment.partition_id,
                node_id: assignment.node_id,
                device_id: assignment.device_id,
                offset_amplitudes: assignment.partition_id * layout.amplitudes_per_partition,
                amplitude_len: layout.amplitudes_per_partition,
                global_qubits: layout.global_qubits,
                local_qubits: layout.local_qubits,
            })
            .collect();

        Ok(DistributedStateHandle {
            state_id,
            dtype,
            scheme: layout.scheme.clone(),
            consumption_mode,
            layout,
            partitions,
        })
    }

    fn plan_round_robin(
        partition_count: usize,
        devices: &[DeviceCapabilities],
    ) -> Vec<PartitionAssignment> {
        (0..partition_count)
            .map(|partition_id| {
                let device = &devices[partition_id % devices.len()];
                PartitionAssignment {
                    partition_id,
                    node_id: device.node_id.clone(),
                    device_id: device.device_id,
                }
            })
            .collect()
    }

    fn plan_weighted(
        partition_count: usize,
        devices: &[DeviceCapabilities],
    ) -> Result<Vec<PartitionAssignment>> {
        let mut weighted_devices = devices
            .iter()
            .map(|device| {
                Ok(WeightedDevice {
                    node_id: device.node_id.clone(),
                    device_id: device.device_id,
                    weight: device.placement_weight()?,
                    assigned: 0,
                    nvlink_peer_count: device.nvlink_peer_count(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if weighted_devices.iter().all(|d| d.weight <= 0.0) {
            return Err(MahoutError::InvalidInput(
                "all devices have non-positive placement weight".to_string(),
            ));
        }

        let mut assignments = Vec::with_capacity(partition_count);
        for partition_id in 0..partition_count {
            let next_idx = weighted_devices
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    let left_score = left.assigned as f64 / left.weight;
                    let right_score = right.assigned as f64 / right.weight;
                    left_score
                        .total_cmp(&right_score)
                        .then_with(|| left.node_id.cmp(&right.node_id))
                        .then_with(|| left.device_id.cmp(&right.device_id))
                })
                .map(|(idx, _)| idx)
                .expect("weighted_devices is non-empty");

            let device = &mut weighted_devices[next_idx];
            device.assigned += 1;
            assignments.push(PartitionAssignment {
                partition_id,
                node_id: device.node_id.clone(),
                device_id: device.device_id,
            });
        }

        Ok(assignments)
    }

    fn plan_topology_aware(
        partition_count: usize,
        devices: &[DeviceCapabilities],
    ) -> Result<Vec<PartitionAssignment>> {
        runtime_profile_scope!("Runtime::PlanPartitions::TopologyAware");

        let mut weighted_devices = devices
            .iter()
            .map(|device| {
                Ok(WeightedDevice {
                    node_id: device.node_id.clone(),
                    device_id: device.device_id,
                    weight: device.placement_weight()?,
                    assigned: 0,
                    nvlink_peer_count: device.nvlink_peer_count(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if weighted_devices.iter().all(|d| d.weight <= 0.0) {
            return Err(MahoutError::InvalidInput(
                "all devices have non-positive placement weight".to_string(),
            ));
        }

        weighted_devices.sort_by(|left, right| {
            right.nvlink_peer_count
                .cmp(&left.nvlink_peer_count)
                .then_with(|| right.weight.total_cmp(&left.weight))
                .then_with(|| left.node_id.cmp(&right.node_id))
                .then_with(|| left.device_id.cmp(&right.device_id))
        });

        let mut assignments = Vec::with_capacity(partition_count);
        for partition_id in 0..partition_count {
            let next_idx = weighted_devices
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    let left_score = left.assigned as f64 / left.weight;
                    let right_score = right.assigned as f64 / right.weight;
                    left_score
                        .total_cmp(&right_score)
                        .then_with(|| right.nvlink_peer_count.cmp(&left.nvlink_peer_count))
                        .then_with(|| left.node_id.cmp(&right.node_id))
                        .then_with(|| left.device_id.cmp(&right.device_id))
                })
                .map(|(idx, _)| idx)
                .expect("weighted_devices is non-empty");

            let device = &mut weighted_devices[next_idx];
            device.assigned += 1;
            assignments.push(PartitionAssignment {
                partition_id,
                node_id: device.node_id.clone(),
                device_id: device.device_id,
            });
        }

        Ok(assignments)
    }
}

#[derive(Clone, Debug)]
struct WeightedDevice {
    node_id: String,
    device_id: usize,
    weight: f64,
    assigned: usize,
    nvlink_peer_count: usize,
}
