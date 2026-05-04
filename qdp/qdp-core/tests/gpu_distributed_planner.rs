use qdp_core::{
    DeviceMesh, DistributionMode, GpuTopology, LinkKind, PlacementPlanner, PlacementRequest,
    ShardPolicy,
};

#[test]
fn placement_planner_emits_contiguous_ranges() {
    let topology = GpuTopology::placeholder(2);
    let mesh = DeviceMesh {
        device_ids: vec![3, 7],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(3, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let plan = PlacementPlanner::plan(&mesh, &request).unwrap();
    assert_eq!(plan.gather_device_id, Some(3));
    assert_eq!(plan.placements.len(), 2);
    assert_eq!(plan.placements[0].device_id, 3);
    assert_eq!(
        (plan.placements[0].start_idx, plan.placements[0].end_idx),
        (0, 4)
    );
    assert_eq!(plan.placements[1].device_id, 7);
    assert_eq!(
        (plan.placements[1].start_idx, plan.placements[1].end_idx),
        (4, 8)
    );
}

#[test]
fn balanced_uneven_policy_supports_non_power_of_two_device_counts() {
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
    let plan = PlacementPlanner::plan(&mesh, &request).unwrap();
    assert_eq!(plan.placements.len(), 3);
    assert_eq!(
        (plan.placements[0].start_idx, plan.placements[0].end_idx),
        (0, 3)
    );
    assert_eq!(
        (plan.placements[1].start_idx, plan.placements[1].end_idx),
        (3, 6)
    );
    assert_eq!(
        (plan.placements[2].start_idx, plan.placements[2].end_idx),
        (6, 8)
    );
}

#[test]
fn equal_policy_rejects_non_power_of_two_device_counts() {
    let topology = GpuTopology::placeholder(3);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1, 2],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(3, DistributionMode::ShardedCapacity, ShardPolicy::Equal);
    let err = PlacementPlanner::plan(&mesh, &request).unwrap_err();
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("power-of-two")
    ));
}

#[test]
fn single_mode_uses_only_first_device() {
    let topology = GpuTopology::placeholder(3);
    let mesh = DeviceMesh {
        device_ids: vec![4, 8, 15],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(3, DistributionMode::Single, ShardPolicy::Equal);
    let plan = PlacementPlanner::plan(&mesh, &request).unwrap();
    assert_eq!(plan.placements.len(), 1);
    assert_eq!(plan.gather_device_id, Some(4));
    assert_eq!(plan.placements[0].device_id, 4);
    assert_eq!((plan.placements[0].start_idx, plan.placements[0].end_idx), (0, 8));
}

#[test]
fn replicated_mode_is_not_implemented() {
    let topology = GpuTopology::placeholder(2);
    let mesh = DeviceMesh {
        device_ids: vec![0, 1],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(2, DistributionMode::Replicated, ShardPolicy::Equal);
    let err = PlacementPlanner::plan(&mesh, &request).unwrap_err();
    assert!(matches!(err, qdp_core::MahoutError::NotImplemented(_)));
}

#[test]
fn sharded_capacity_prefers_topology_recommended_device_order() {
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
    let mesh = DeviceMesh {
        device_ids: vec![10, 11, 12],
        devices: Vec::new(),
        topology,
    };
    let request = PlacementRequest::new(
        3,
        DistributionMode::ShardedCapacity,
        ShardPolicy::BalancedUneven,
    );

    let plan = PlacementPlanner::plan(&mesh, &request).unwrap();
    let ordered_ids = plan
        .placements
        .iter()
        .map(|placement| placement.device_id)
        .collect::<Vec<_>>();

    assert_eq!(plan.gather_device_id, Some(11));
    assert_eq!(ordered_ids, vec![11, 10, 12]);
}
