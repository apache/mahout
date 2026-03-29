use qdp_runtime::{
    ConsumptionMode, Coordinator, DType, DeviceCapabilities, GatherTarget, HostPlatform,
    InProcessWorker, PlacementPolicy, RuntimeJobSpec, WorkerRegistration,
};

fn worker(worker_id: &str, node_id: &str, device_id: usize) -> InProcessWorker {
    InProcessWorker::new(WorkerRegistration {
        worker_id: worker_id.to_string(),
        node_id: node_id.to_string(),
        devices: vec![DeviceCapabilities {
            node_id: node_id.to_string(),
            device_id,
            device_name: format!("mock-gpu-{}", device_id),
            total_memory_bytes: 48 * 1024 * 1024 * 1024,
            free_memory_bytes: 40 * 1024 * 1024 * 1024,
            max_safe_allocation_bytes: 40 * 1024 * 1024 * 1024,
            measured_encode_samples_per_sec: Some(3000.0),
            host_platform: HostPlatform::Linux,
            stability_factor: 1.0,
            peer_links: Vec::new(),
        }],
    })
    .expect("valid worker registration")
}

fn main() {
    let worker_a = worker("worker-a", "node-a", 0);
    let worker_b = worker("worker-b", "node-b", 0);

    let mut coordinator = Coordinator::new();
    coordinator
        .register_worker(&worker_a)
        .expect("register worker-a");
    coordinator
        .register_worker(&worker_b)
        .expect("register worker-b");

    coordinator
        .plan_job(RuntimeJobSpec {
            job_id: "smoke-job".to_string(),
            state_id: "smoke-state".to_string(),
            global_qubits: 5,
            dtype: DType::Complex64,
            consumption_mode: ConsumptionMode::GatherFullState,
            placement_policy: PlacementPolicy::Weighted,
        })
        .expect("plan job");

    let completed = coordinator
        .run_job_with_workers("smoke-job", &[worker_a, worker_b])
        .expect("run job");

    println!("job_status={:?}", completed.status);
    println!("partition_count={}", completed.state.layout.partition_count);
    println!("objects={}", coordinator.objects_for_job("smoke-job").len());

    let gather = coordinator
        .build_gather_plan("smoke-job", GatherTarget::HostMemory)
        .expect("gather plan");
    println!("gather_segments={}", gather.segments.len());
    for segment in gather.segments {
        println!(
            "partition={} source={}/{} offset={} len={} handle={}",
            segment.partition_id,
            segment.source_node_id,
            segment.source_device_id,
            segment.destination_offset_amplitudes,
            segment.amplitude_len,
            segment.source_storage_handle
        );
    }
}
