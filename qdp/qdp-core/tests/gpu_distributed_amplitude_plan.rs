use qdp_core::{
    DeviceMesh, DistributedAmplitudePlan, DistributedStateLayout, DistributionMode, GpuTopology,
    PlacementRequest, Precision, ShardPolicy,
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
fn distributed_amplitude_plan_rejects_global_qubits_when_local_shard_exceeds_single_gpu_limit() {
    let topology = GpuTopology::placeholder(1);
    let mesh = DeviceMesh {
        device_ids: vec![0],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(31, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let err = DistributedAmplitudePlan::for_request(&mesh, request).unwrap_err();
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg) if msg.contains("per-device capacity")
    ));
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

    let err = match DistributedStateLayout::new(&mesh, &plan, Precision::Float64) {
        Ok(_) => panic!("expected malformed mesh to be rejected"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("device handles")
    ));
}
