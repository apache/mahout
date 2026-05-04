use qdp_core::gpu::{Communicator, HostCommunicator};

#[test]
fn host_communicator_reduce_sum_returns_total() {
    let comm = HostCommunicator;
    let values = vec![1.0, 2.0, 3.0];
    assert_eq!(comm.reduce_sum_f64(&values).unwrap(), 6.0);
}

#[test]
fn host_communicator_reduce_sum_rejects_empty_inputs() {
    let comm = HostCommunicator;
    let err = comm.reduce_sum_f64(&[]).unwrap_err();
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("at least one value")
    ));
}
