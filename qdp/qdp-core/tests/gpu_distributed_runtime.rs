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

use std::sync::{Arc, Mutex};

use qdp_core::gpu::{
    CollectiveCommunicator, DeviceMesh, DistributedExecutionContext, GpuTopology,
    LocalCollectiveCommunicator,
};
use qdp_core::{DistributionMode, PlacementRequest, Precision, QdpEngine, ShardPolicy};

#[derive(Clone, Copy)]
struct TestCollective {
    rank: usize,
    world_size: usize,
}

impl CollectiveCommunicator for TestCollective {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce_sum_f64(&self, local_value: f64) -> qdp_core::Result<f64> {
        Ok(local_value)
    }
}

#[derive(Clone)]
struct RecordingCollective {
    rank: usize,
    world_size: usize,
    global_sum: f64,
    seen: Arc<Mutex<Vec<f64>>>,
}

impl CollectiveCommunicator for RecordingCollective {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce_sum_f64(&self, local_value: f64) -> qdp_core::Result<f64> {
        self.seen
            .lock()
            .expect("recording collective observations mutex poisoned")
            .push(local_value);
        Ok(self.global_sum)
    }
}

#[test]
fn rank_local_execution_context_reports_rank_metadata() {
    let collectives = TestCollective {
        rank: 1,
        world_size: 3,
    };
    let mesh = DeviceMesh {
        device_ids: vec![0],
        devices: Vec::new(),
        topology: GpuTopology::placeholder(1),
    };
    let execution =
        DistributedExecutionContext::rank_local_with_mesh(1, 3, mesh, &collectives).unwrap();

    assert_eq!(execution.rank(), 1);
    assert_eq!(execution.world_size(), 3);
    assert_eq!(execution.mesh().num_devices(), 1);
    assert!(execution.device_collectives().is_none());
}

#[test]
fn rank_local_execution_context_rejects_collective_metadata_mismatch() {
    let collectives = LocalCollectiveCommunicator;
    let mesh = DeviceMesh {
        device_ids: vec![0],
        devices: Vec::new(),
        topology: GpuTopology::placeholder(1),
    };

    let err =
        DistributedExecutionContext::rank_local_with_mesh(1, 3, mesh, &collectives).unwrap_err();

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("collective") && msg.contains("rank")
    ));
}

#[test]
fn rank_local_execution_context_rejects_rank_out_of_range() {
    let collectives = LocalCollectiveCommunicator;
    let err = DistributedExecutionContext::rank_local(2, 2, vec![0], &collectives).unwrap_err();

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("rank") && msg.contains("world size")
    ));
}

#[test]
fn single_process_rejects_collective_metadata_mismatch_before_cuda_init() {
    let collectives = TestCollective {
        rank: 1,
        world_size: 3,
    };
    let err = DistributedExecutionContext::single_process(vec![0], &collectives).unwrap_err();

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("collective") && msg.contains("rank")
    ));
}

#[test]
fn prepare_distributed_amplitude_handles_padding_tail_in_norm() {
    #[cfg(target_os = "linux")]
    if cudarc::driver::CudaDevice::new(0).is_err() {
        return;
    }

    let prepared = QdpEngine::prepare_distributed_amplitude(
        vec![0],
        &[1.0, 2.0, 3.0],
        2,
        Precision::Float64,
        None,
    )
    .unwrap();

    let expected = 1.0 / 14.0f64.sqrt();
    assert!((prepared.inv_norm - expected).abs() < 1e-12);
}

#[test]
fn prepare_distributed_amplitude_accepts_custom_request() {
    #[cfg(target_os = "linux")]
    if cudarc::driver::CudaDevice::new(0).is_err() {
        return;
    }
    #[cfg(target_os = "linux")]
    if cudarc::driver::CudaDevice::new(1).is_err() {
        return;
    }

    let prepared = QdpEngine::prepare_distributed_amplitude(
        vec![0, 1],
        &[1.0, 2.0, 3.0],
        2,
        Precision::Float32,
        Some(PlacementRequest::new(
            2,
            DistributionMode::ShardedCapacity,
            ShardPolicy::Equal,
        )),
    )
    .unwrap();

    assert_eq!(prepared.plan.num_qubits, 2);
    assert_eq!(prepared.plan.placement.num_devices(), 2);
}

#[test]
fn prepare_distributed_amplitude_rejects_request_qubit_mismatch() {
    let err = match QdpEngine::prepare_distributed_amplitude(
        vec![0],
        &[1.0, 2.0],
        2,
        Precision::Float64,
        Some(PlacementRequest::new(
            3,
            DistributionMode::ShardedCapacity,
            ShardPolicy::Equal,
        )),
    ) {
        Ok(_) => panic!("expected qubit mismatch to be rejected"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("qubit mismatch")
    ));
}

#[test]
fn prepare_distributed_amplitude_rejects_empty_input() {
    let err =
        match QdpEngine::prepare_distributed_amplitude(vec![0], &[], 1, Precision::Float64, None) {
            Ok(_) => panic!("expected empty input to be rejected"),
            Err(err) => err,
        };

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("cannot be empty")
    ));
}

#[test]
fn prepare_distributed_amplitude_rejects_zero_norm_input() {
    let mesh = DeviceMesh {
        device_ids: vec![0],
        devices: Vec::new(),
        topology: GpuTopology::placeholder(1),
    };
    let collectives = TestCollective {
        rank: 0,
        world_size: 1,
    };
    let execution =
        DistributedExecutionContext::rank_local_with_mesh(0, 1, mesh, &collectives).unwrap();
    let err = match QdpEngine::prepare_distributed_amplitude_on(
        &execution,
        &[0.0, 0.0],
        1,
        Precision::Float64,
        None,
    ) {
        Ok(_) => panic!("expected zero-norm input to be rejected"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("zero or non-finite norm")
    ));
}

#[test]
fn prepare_distributed_amplitude_rejects_non_finite_input() {
    let mesh = DeviceMesh {
        device_ids: vec![0],
        devices: Vec::new(),
        topology: GpuTopology::placeholder(1),
    };
    let collectives = TestCollective {
        rank: 0,
        world_size: 1,
    };
    let execution =
        DistributedExecutionContext::rank_local_with_mesh(0, 1, mesh, &collectives).unwrap();
    let err = match QdpEngine::prepare_distributed_amplitude_on(
        &execution,
        &[1.0, f64::NAN],
        1,
        Precision::Float64,
        None,
    ) {
        Ok(_) => panic!("expected non-finite input to be rejected"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("zero or non-finite norm")
    ));
}

#[test]
fn prepare_distributed_amplitude_reduces_before_rejecting_non_finite_input() {
    let mesh = DeviceMesh {
        device_ids: vec![0],
        devices: Vec::new(),
        topology: GpuTopology::placeholder(1),
    };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let collectives = RecordingCollective {
        rank: 0,
        world_size: 1,
        global_sum: f64::NAN,
        seen: Arc::clone(&seen),
    };
    let execution =
        DistributedExecutionContext::rank_local_with_mesh(0, 1, mesh, &collectives).unwrap();

    let err = match QdpEngine::prepare_distributed_amplitude_on(
        &execution,
        &[1.0, f64::NAN],
        1,
        Precision::Float64,
        None,
    ) {
        Ok(_) => panic!("expected non-finite input to be rejected after reduction"),
        Err(err) => err,
    };

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(!seen[0].is_finite());
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("zero or non-finite norm")
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
        Ok(_) => panic!("expected invalid input to be rejected before mesh creation"),
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
fn prepare_distributed_amplitude_uses_only_rank_local_norm_contribution() {
    let device0 = match cudarc::driver::CudaDevice::new(0) {
        Ok(device) => device,
        Err(_) => return,
    };
    let device1 = match cudarc::driver::CudaDevice::new(1) {
        Ok(device) => device,
        Err(_) => return,
    };
    let mesh = qdp_core::gpu::DeviceMesh::from_parts(
        vec![0, 1],
        vec![device0, device1],
        qdp_core::gpu::GpuTopology::placeholder(2),
    )
    .unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let collectives = RecordingCollective {
        rank: 1,
        world_size: 2,
        global_sum: 30.0,
        seen: Arc::clone(&seen),
    };
    let execution =
        DistributedExecutionContext::rank_local_with_mesh(1, 2, mesh, &collectives).unwrap();
    let request = PlacementRequest::new_with_world(
        2,
        DistributionMode::ShardedCapacity,
        ShardPolicy::Equal,
        2,
    )
    .unwrap();

    let prepared = QdpEngine::prepare_distributed_amplitude_on(
        &execution,
        &[1.0, 2.0, 3.0, 4.0],
        2,
        Precision::Float64,
        Some(request),
    )
    .unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![20.0]);
    assert!((prepared.inv_norm - (1.0 / 30.0f64.sqrt())).abs() < 1e-12);
    assert_eq!(prepared.layout.rank_id, 1);
    assert_eq!(prepared.layout.num_shards(), 2);
    assert_eq!(prepared.layout.shards[0].shard_id, 1);
    assert_eq!(prepared.layout.shards[1].shard_id, 3);
}

#[test]
fn prepare_distributed_amplitude_on_rejects_request_world_mismatch() {
    let mesh = qdp_core::gpu::DeviceMesh {
        device_ids: vec![0],
        devices: Vec::new(),
        topology: qdp_core::gpu::GpuTopology::placeholder(1),
    };
    let collectives = TestCollective {
        rank: 0,
        world_size: 2,
    };
    let execution =
        DistributedExecutionContext::rank_local_with_mesh(0, 2, mesh, &collectives).unwrap();
    let request = PlacementRequest::new(2, DistributionMode::ShardedCapacity, ShardPolicy::Equal);

    let err = match QdpEngine::prepare_distributed_amplitude_on(
        &execution,
        &[1.0, 2.0],
        2,
        Precision::Float64,
        Some(request),
    ) {
        Ok(_) => panic!("expected request world mismatch to be rejected"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("world size")
    ));
}

#[test]
fn prepare_distributed_amplitude_on_defaults_request_world_to_execution_world() {
    let mesh = qdp_core::gpu::DeviceMesh {
        device_ids: vec![0, 1],
        devices: Vec::new(),
        topology: qdp_core::gpu::GpuTopology::placeholder(2),
    };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let collectives = RecordingCollective {
        rank: 1,
        world_size: 2,
        global_sum: 30.0,
        seen: Arc::clone(&seen),
    };
    let execution =
        DistributedExecutionContext::rank_local_with_mesh(1, 2, mesh, &collectives).unwrap();

    let err = match QdpEngine::prepare_distributed_amplitude_on(
        &execution,
        &[1.0, 2.0, 3.0, 4.0],
        2,
        Precision::Float64,
        None,
    ) {
        Ok(_) => panic!("expected handle-less mesh to fail after request resolution"),
        Err(err) => err,
    };

    assert_eq!(*seen.lock().unwrap(), vec![20.0]);
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("Device mesh / device handles mismatch")
    ));
}

#[test]
fn prepare_distributed_amplitude_on_plans_from_rank_local_mesh() {
    let mesh = qdp_core::gpu::DeviceMesh {
        device_ids: vec![0],
        devices: Vec::new(),
        topology: qdp_core::gpu::GpuTopology::placeholder(1),
    };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let collectives = RecordingCollective {
        rank: 1,
        world_size: 2,
        global_sum: 30.0,
        seen: Arc::clone(&seen),
    };
    let execution =
        DistributedExecutionContext::rank_local_with_mesh(1, 2, mesh, &collectives).unwrap();
    let request = PlacementRequest::new_with_world(
        2,
        DistributionMode::ShardedCapacity,
        ShardPolicy::Equal,
        2,
    )
    .unwrap();

    let err = match QdpEngine::prepare_distributed_amplitude_on(
        &execution,
        &[1.0, 2.0, 3.0, 4.0],
        2,
        Precision::Float64,
        Some(request),
    ) {
        Ok(_) => panic!("expected handle-less mesh to fail after rank-local planning"),
        Err(err) => err,
    };

    assert_eq!(*seen.lock().unwrap(), vec![25.0]);
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("Device mesh / device handles mismatch")
    ));
}
