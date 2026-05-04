use qdp_core::{DistributionMode, PlacementRequest, Precision, QdpEngine, ShardPolicy};

#[test]
fn prepare_distributed_amplitude_handles_padding_tail_in_norm() {
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
    let err = match QdpEngine::prepare_distributed_amplitude(
        vec![0],
        &[],
        1,
        Precision::Float64,
        None,
    ) {
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
    let err = match QdpEngine::prepare_distributed_amplitude(
        vec![0],
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
    let err = match QdpEngine::prepare_distributed_amplitude(
        vec![0],
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
        if msg.contains("NaN or Inf")
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
