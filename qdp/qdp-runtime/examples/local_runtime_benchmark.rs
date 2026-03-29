use std::time::Instant;

use qdp_runtime::{
    ConsumptionMode, Coordinator, DType, DeviceCapabilities, HostPlatform, InProcessWorker,
    PlacementPolicy, RuntimeJobSpec, WorkerRegistration,
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

    let plan_start = Instant::now();
    coordinator
        .plan_job(RuntimeJobSpec {
            job_id: "bench-job".to_string(),
            state_id: "bench-state".to_string(),
            global_qubits: 16,
            dtype: DType::Complex64,
            consumption_mode: ConsumptionMode::GatherFullState,
            placement_policy: PlacementPolicy::Weighted,
        })
        .expect("plan job");
    let plan_elapsed = plan_start.elapsed();

    let run_start = Instant::now();
    let completed = coordinator
        .run_job_with_workers("bench-job", &[worker_a, worker_b])
        .expect("run job");
    let run_elapsed = run_start.elapsed();

    println!("job_status={:?}", completed.status);
    println!("planning_ms={:.3}", plan_elapsed.as_secs_f64() * 1000.0);
    println!("execution_ms={:.3}", run_elapsed.as_secs_f64() * 1000.0);
    println!("objects={}", coordinator.objects_for_job("bench-job").len());
    println!("partitions={}", completed.tasks.len());
}
