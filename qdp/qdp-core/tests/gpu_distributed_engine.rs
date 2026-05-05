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

use qdp_core::gpu::{
    DeviceMesh, DistributedAmplitudePlan, DistributedExecutionContext, DistributedStateLayout,
    DistributionMode, GpuTopology, LinkKind, LocalCollectiveCommunicator, PlacementRequest,
    QuantumEncoder, ShardPolicy,
};
use qdp_core::{Precision, QdpEngine};

mod common;

#[test]
fn prepare_distributed_amplitude_returns_expected_metadata() {
    #[cfg(target_os = "linux")]
    if cudarc::driver::CudaDevice::new(1).is_err() {
        return;
    }

    let prepared = QdpEngine::prepare_distributed_amplitude(
        vec![0, 1],
        &[1.0, 2.0, 3.0],
        2,
        Precision::Float32,
        None,
    )
    .unwrap();

    assert_eq!(prepared.mesh.num_devices(), 2);
    assert_eq!(prepared.plan.uniform_shard_len, Some(2));
    assert_eq!(prepared.layout.num_shards(), 2);
    assert_eq!(prepared.layout.global_len, 4);

    let expected = 1.0 / 14.0f64.sqrt();
    assert!((prepared.inv_norm - expected).abs() < 1e-12);
}

#[test]
fn prepare_distributed_amplitude_on_execution_context_returns_expected_metadata() {
    #[cfg(target_os = "linux")]
    if cudarc::driver::CudaDevice::new(1).is_err() {
        return;
    }

    let collectives = LocalCollectiveCommunicator;
    let execution = DistributedExecutionContext::single_process(vec![0, 1], &collectives).unwrap();
    let prepared = QdpEngine::prepare_distributed_amplitude_on(
        &execution,
        &[1.0, 2.0, 3.0],
        2,
        Precision::Float32,
        None,
    )
    .unwrap();

    assert_eq!(prepared.mesh.num_devices(), 2);
    assert_eq!(prepared.plan.uniform_shard_len, Some(2));
    assert_eq!(prepared.layout.num_shards(), 2);
    assert_eq!(prepared.layout.global_len, 4);
}

#[cfg(target_os = "linux")]
fn reordered_three_device_mesh() -> Option<DeviceMesh> {
    let device0 = cudarc::driver::CudaDevice::new(0).ok()?;
    let device1 = cudarc::driver::CudaDevice::new(1).ok()?;
    let device2 = cudarc::driver::CudaDevice::new(2).ok()?;
    let topology = GpuTopology {
        peer_access: vec![
            vec![true, true, false],
            vec![true, true, true],
            vec![false, true, true],
        ],
        links: vec![
            vec![LinkKind::SameDevice, LinkKind::Pix, LinkKind::Unknown],
            vec![LinkKind::Pix, LinkKind::SameDevice, LinkKind::Node],
            vec![LinkKind::Unknown, LinkKind::Node, LinkKind::SameDevice],
        ],
    };

    DeviceMesh::from_parts(vec![0, 1, 2], vec![device0, device1, device2], topology).ok()
}

#[test]
#[cfg(target_os = "linux")]
fn distributed_layout_uses_device_handles_for_reordered_placements() {
    let Some(mesh) = reordered_three_device_mesh() else {
        return;
    };

    let request = PlacementRequest::new(
        2,
        DistributionMode::ShardedCapacity,
        ShardPolicy::BalancedUneven,
    );
    let plan = DistributedAmplitudePlan::for_request(&mesh, request).unwrap();
    assert_eq!(plan.placement.placements[0].device_id, 1);

    let layout = DistributedStateLayout::new(&mesh, &plan, Precision::Float32).unwrap();
    assert_eq!(layout.shards[0].device_id, 1);
    assert_eq!(layout.shards[0].device.ordinal(), 1);
    assert_eq!(layout.shards[1].device_id, 0);
    assert_eq!(layout.shards[1].device.ordinal(), 0);
    assert_eq!(layout.shards[2].device_id, 2);
    assert_eq!(layout.shards[2].device.ordinal(), 2);
}

#[test]
#[cfg(target_os = "linux")]
fn distributed_encoding_uses_device_handles_for_reordered_placements() {
    let Some(mesh) = reordered_three_device_mesh() else {
        return;
    };
    let device0 = mesh.device_for_id(0).unwrap();

    let collectives = LocalCollectiveCommunicator;
    let execution = DistributedExecutionContext::new(mesh, &collectives);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let single_state = qdp_core::gpu::AmplitudeEncoder
        .encode(&device0, &input, 3)
        .unwrap();
    let single_host = single_state
        .copy_to_host_f64(&device0)
        .unwrap()
        .into_iter()
        .map(|value| value.x)
        .collect::<Vec<_>>();

    let state = QdpEngine::encode_distributed_amplitude_to_shards_on(
        &execution,
        &input,
        3,
        Precision::Float64,
        Some(PlacementRequest::new(
            3,
            DistributionMode::ShardedCapacity,
            ShardPolicy::BalancedUneven,
        )),
    )
    .unwrap();

    assert_eq!(state.shards[0].device_id, 1);
    assert_eq!(state.shards[0].device.ordinal(), 1);
    assert_eq!(state.shards[1].device_id, 0);
    assert_eq!(state.shards[1].device.ordinal(), 0);
    assert_eq!(state.shards[2].device_id, 2);
    assert_eq!(state.shards[2].device.ordinal(), 2);

    let mut distributed_host = Vec::new();
    for shard_id in 0..state.num_shards() {
        distributed_host.extend(
            state
                .copy_shard_to_host_f64(shard_id)
                .unwrap()
                .into_iter()
                .map(|value| value.x),
        );
    }

    assert_eq!(distributed_host, single_host);
}

#[test]
#[cfg(target_os = "linux")]
fn encode_distributed_amplitude_to_shards_returns_real_buffers() {
    let Some(device0) = common::cuda_device() else {
        return;
    };
    let device1 = match cudarc::driver::CudaDevice::new(1) {
        Ok(device) => device,
        Err(_) => return,
    };

    let _ = device1;

    let state = QdpEngine::encode_distributed_amplitude_to_shards(
        vec![0, 1],
        &[1.0, 2.0, 3.0],
        2,
        Precision::Float64,
        None,
    )
    .unwrap();

    assert_eq!(state.num_shards(), 2);

    let shard0 = state.copy_shard_to_host_f64(0).unwrap();
    let shard1 = state.copy_shard_to_host_f64(1).unwrap();

    let expected = 1.0 / 14.0f64.sqrt();
    let shard0 = shard0.iter().map(|value| value.x).collect::<Vec<_>>();
    let shard1 = shard1.iter().map(|value| value.x).collect::<Vec<_>>();

    assert!((shard0[0] - expected).abs() < 1e-12);
    assert!((shard0[1] - (2.0 * expected)).abs() < 1e-12);
    assert!((shard1[0] - (3.0 * expected)).abs() < 1e-12);
    assert_eq!(shard1[1], 0.0);

    let _ = device0;
}

#[test]
#[cfg(target_os = "linux")]
fn distributed_amplitude_matches_single_gpu_reference_on_two_gpus() {
    let device0 = match cudarc::driver::CudaDevice::new(0) {
        Ok(device) => device,
        Err(_) => return,
    };
    let device1 = match cudarc::driver::CudaDevice::new(1) {
        Ok(device) => device,
        Err(_) => return,
    };

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let encoder = qdp_core::gpu::AmplitudeEncoder;
    let single_state = encoder.encode(&device0, &input, 3).unwrap();
    let single_host = single_state
        .copy_to_host_f64(&device0)
        .unwrap()
        .into_iter()
        .map(|value| value.x)
        .collect::<Vec<_>>();

    let distributed = QdpEngine::encode_distributed_amplitude_to_shards(
        vec![0, 1],
        &input,
        3,
        Precision::Float64,
        None,
    )
    .unwrap();

    let mut distributed_host = Vec::new();
    distributed_host.extend(
        distributed
            .copy_shard_to_host_f64(0)
            .unwrap()
            .into_iter()
            .map(|value| value.x),
    );
    distributed_host.extend(
        distributed
            .copy_shard_to_host_f64(1)
            .unwrap()
            .into_iter()
            .map(|value| value.x),
    );

    assert_eq!(distributed_host.len(), single_host.len());
    for (idx, (distributed_value, single_value)) in
        distributed_host.iter().zip(single_host.iter()).enumerate()
    {
        assert!(
            (*distributed_value - *single_value).abs() < 1e-12,
            "Mismatch at amplitude {}: distributed={} single={}",
            idx,
            distributed_value,
            single_value
        );
    }

    let _ = device1;
}

#[test]
fn prepare_distributed_amplitude_rejects_oversized_inputs() {
    let err = match QdpEngine::prepare_distributed_amplitude(
        vec![0],
        &[1.0, 2.0, 3.0],
        1,
        Precision::Float64,
        None,
    ) {
        Ok(_) => panic!("expected oversized input to be rejected"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("exceeds state vector size")
    ));
}

#[test]
fn prepare_distributed_amplitude_validates_input_before_building_mesh() {
    let err = match QdpEngine::prepare_distributed_amplitude(
        vec![9999],
        &[],
        1,
        Precision::Float64,
        None,
    ) {
        Ok(_) => panic!("expected empty input to be rejected before mesh creation"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("cannot be empty")
    ));
}

#[test]
#[cfg(target_os = "linux")]
fn distributed_state_rejects_out_of_range_shard_reads() {
    let Some(_device0) = common::cuda_device() else {
        return;
    };

    let state = QdpEngine::encode_distributed_amplitude_to_shards(
        vec![0],
        &[1.0, 2.0],
        1,
        Precision::Float64,
        None,
    )
    .unwrap();

    let err = state.copy_shard_to_host_f64(1).unwrap_err();
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("out of range")
    ));
}

#[test]
#[cfg(target_os = "linux")]
fn distributed_state_rejects_precision_mismatched_host_reads() {
    let Some(_device0) = common::cuda_device() else {
        return;
    };

    let state = QdpEngine::encode_distributed_amplitude_to_shards(
        vec![0],
        &[1.0, 2.0],
        1,
        Precision::Float32,
        None,
    )
    .unwrap();

    let err = state.copy_shard_to_host_f64(0).unwrap_err();
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("float32 data, not float64")
    ));
}
