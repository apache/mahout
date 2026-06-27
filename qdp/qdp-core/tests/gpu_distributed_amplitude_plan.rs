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

use qdp_core::Precision;
use qdp_core::gpu::{
    DeviceMesh, DistributedAmplitudePlan, DistributedStateLayout, DistributedStatePlan,
    DistributionMode, GpuTopology, PlacementPlanner, PlacementRequest, ShardPolicy,
};

#[test]
fn distributed_amplitude_plan_has_expected_shard_math() {
    let topology = GpuTopology::placeholder(4);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1, 2, 3],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(4, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let plan = DistributedAmplitudePlan::for_request(&mesh, request).unwrap();
    assert_eq!(plan.global_len, 16);
    assert_eq!(plan.shard_bits, Some(2));
    assert_eq!(plan.uniform_shard_len, Some(4));
    assert_eq!(plan.shard_range(0).unwrap(), (0, 4));
    assert_eq!(plan.shard_range(3).unwrap(), (12, 16));
}

#[test]
fn distributed_state_plan_carries_workload_neutral_shard_math() {
    let topology = GpuTopology::placeholder(4);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1, 2, 3],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(4, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let placement = PlacementPlanner::plan(&mesh, &request).unwrap();
    let plan = DistributedStatePlan::from_placement(request, placement).unwrap();

    assert_eq!(plan.request().num_qubits, 4);
    assert_eq!(plan.placement().num_devices(), 4);
    assert_eq!(plan.global_len, 16);
    assert_eq!(plan.num_devices, 4);
    assert_eq!(plan.shard_bits, Some(2));
    assert_eq!(plan.uniform_shard_len, Some(4));
    assert_eq!(plan.shard_range(2).unwrap(), (8, 12));
    assert_eq!(
        plan.estimated_max_shard_bytes(Precision::Float32).unwrap(),
        32
    );
}

#[test]
fn distributed_state_plan_exposes_rank_local_placements() {
    let topology = GpuTopology::placeholder(2);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new_with_world(
        4,
        DistributionMode::ShardedCapacity,
        ShardPolicy::Equal,
        2,
    )
    .unwrap();
    let placement = PlacementPlanner::plan_rank_local(&mesh, &request).unwrap();
    let plan = DistributedStatePlan::from_placement(request, placement).unwrap();

    let rank_one = plan.placements_for_rank_iter(1).collect::<Vec<_>>();

    assert_eq!(rank_one.len(), 2);
    assert_eq!(rank_one[0].rank_id, 1);
    assert_eq!(rank_one[0].device_id, 0);
    assert_eq!((rank_one[0].start_idx, rank_one[0].end_idx), (4, 8));
    assert_eq!(rank_one[1].rank_id, 1);
    assert_eq!(rank_one[1].device_id, 1);
    assert_eq!((rank_one[1].start_idx, rank_one[1].end_idx), (12, 16));
}

#[test]
fn distributed_amplitude_plan_exposes_shared_state_plan() {
    let topology = GpuTopology::placeholder(2);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(3, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let plan = DistributedAmplitudePlan::for_request(&mesh, request).unwrap();

    let shared: &DistributedStatePlan = plan.as_ref();
    assert_eq!(shared.global_len, plan.global_len);
    assert_eq!(shared.shard_range(1).unwrap(), plan.shard_range(1).unwrap());
}

#[test]
fn distributed_amplitude_plan_rejects_too_many_devices_for_qubits() {
    let topology = GpuTopology::placeholder(4);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1, 2, 3],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(1, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let err = DistributedAmplitudePlan::for_request(&mesh, request).unwrap_err();
    assert!(matches!(err, qdp_core::MahoutError::InvalidInput(_)));
}

#[test]
fn distributed_amplitude_plan_allows_extra_global_qubit_when_shards_fit() {
    let topology = GpuTopology::placeholder(2);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(31, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let plan = DistributedAmplitudePlan::for_request(&mesh, request).unwrap();
    assert_eq!(plan.global_len, 1usize << 31);
    assert_eq!(plan.uniform_shard_len, Some(1usize << 30));
}

#[test]
fn distributed_amplitude_plan_supports_q34_with_balanced_six_gpu_shards() {
    let topology = GpuTopology::placeholder(6);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1, 2, 3, 4, 5],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(
        34,
        DistributionMode::ShardedCapacity,
        ShardPolicy::BalancedUneven,
    );
    let plan = DistributedAmplitudePlan::for_request(&mesh, request).unwrap();

    assert_eq!(plan.global_len, 1usize << 34);
    assert_eq!(plan.num_devices, 6);
    assert_eq!(plan.shard_bits, None);
    assert_eq!(plan.uniform_shard_len, None);
    assert_eq!(plan.shard_range(0).unwrap(), (0, 2_863_311_531));
    assert_eq!(
        plan.shard_range(5).unwrap(),
        (14_316_557_654, 17_179_869_184)
    );
}

#[test]
fn distributed_amplitude_plan_rejects_zero_qubits() {
    let topology = GpuTopology::placeholder(1);
    let mesh = DeviceMesh {
        device_ids: vec![0],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(0, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let err = DistributedAmplitudePlan::for_request(&mesh, request).unwrap_err();
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("at least 1")
    ));
}

#[test]
fn distributed_amplitude_plan_reports_out_of_range_shard_ids() {
    let topology = GpuTopology::placeholder(2);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(2, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let plan = DistributedAmplitudePlan::for_request(&mesh, request).unwrap();
    let err = plan.shard_range(2).unwrap_err();
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("out of range")
    ));
}

#[test]
fn balanced_uneven_distributed_amplitude_plan_omits_uniform_shard_metadata() {
    let topology = GpuTopology::placeholder(3);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1, 2],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(
        3,
        DistributionMode::ShardedCapacity,
        ShardPolicy::BalancedUneven,
    );
    let plan = DistributedAmplitudePlan::for_request(&mesh, request).unwrap();
    assert_eq!(plan.shard_bits, None);
    assert_eq!(plan.uniform_shard_len, None);
    assert_eq!(plan.shard_range(2).unwrap(), (6, 8));
}

#[test]
fn distributed_state_layout_rejects_mesh_with_missing_device_handles() {
    let topology = GpuTopology::placeholder(2);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(2, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let plan = DistributedAmplitudePlan::for_request(&mesh, request).unwrap();

    let err = match DistributedStateLayout::new_for_rank(&mesh, &plan, Precision::Float64, 0) {
        Ok(_) => panic!("expected malformed mesh to be rejected"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("device handles")
    ));
}

#[test]
fn distributed_state_layout_rejects_rank_out_of_range_before_device_handles() {
    let mesh = DeviceMesh {
        device_ids: vec![1],
        devices: Vec::new(),
        topology: GpuTopology::placeholder(1),
    };
    let planning_mesh = DeviceMesh {
        device_ids: vec![0, 1],
        devices: Vec::new(),
        topology: GpuTopology::placeholder(2),
    };
    let request = PlacementRequest::new_with_world(
        2,
        DistributionMode::ShardedCapacity,
        ShardPolicy::Equal,
        2,
    )
    .unwrap();
    let plan = DistributedAmplitudePlan::for_request(&planning_mesh, request).unwrap();

    let err =
        DistributedStateLayout::new_for_rank(&mesh, &plan, Precision::Float64, 2).unwrap_err();

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("rank") && msg.contains("world size")
    ));
}

#[test]
#[cfg(target_os = "linux")]
fn distributed_state_layout_can_be_rank_local() {
    let device1 = match cudarc::driver::CudaDevice::new(1) {
        Ok(device) => device,
        Err(_) => return,
    };
    let rank_mesh =
        DeviceMesh::from_parts(vec![1], vec![device1], GpuTopology::placeholder(1)).unwrap();
    let planning_mesh = DeviceMesh {
        device_ids: vec![0, 1],
        devices: Vec::new(),
        topology: GpuTopology::placeholder(2),
    };
    let request = PlacementRequest::new_with_world(
        2,
        DistributionMode::ShardedCapacity,
        ShardPolicy::Equal,
        2,
    )
    .unwrap();
    let plan = DistributedAmplitudePlan::for_request(&planning_mesh, request).unwrap();

    let layout =
        DistributedStateLayout::new_for_rank(&rank_mesh, &plan, Precision::Float64, 1).unwrap();

    assert_eq!(layout.rank_id, 1);
    assert_eq!(layout.world_size, 2);
    assert_eq!(layout.num_shards(), 1);
    assert_eq!(layout.shards().len(), 1);
    assert_eq!(layout.iter_shards().count(), 1);
    assert_eq!(layout.shards()[0].rank_id, 1);
    assert_eq!(layout.shards()[0].device_id, 1);
    assert_eq!(layout.shards()[0].shard_id, 1);
}

#[test]
fn distributed_amplitude_plan_reports_rank_local_ranges() {
    let topology = GpuTopology::placeholder(4);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1, 2, 3],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new_with_world(
        4,
        DistributionMode::ShardedCapacity,
        ShardPolicy::Equal,
        2,
    )
    .unwrap();
    let plan = DistributedAmplitudePlan::for_request(&mesh, request).unwrap();

    assert_eq!(plan.num_local_shards(0), 2);
    assert_eq!(plan.num_local_shards(1), 2);
    assert_eq!(plan.rank_shard_ranges(1), vec![(4, 8), (12, 16)]);
}
