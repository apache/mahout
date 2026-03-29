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

#[macro_use]
mod profiling;

pub mod coordinator;
pub mod model;
pub mod planning;
pub mod topology;
pub mod worker;

pub use coordinator::Coordinator;
pub use model::{
    ConsumptionMode, DType, DistributedStateHandle, GatherPlan, GatherSegment, GatherTarget,
    JobStatus, MetricContribution, MetricReduceOp, PartitionAssignment, PartitionInput,
    PartitionLayout, PartitionScheme, PartitionTask, PartitionTaskResult, PlannedJob,
    ReducePlan, ReducedMetricValue, RetryPolicy, RuntimeJobSpec, RuntimeObjectKind,
    RuntimeObjectLocation, RuntimeObjectRecord, StatePartitionRef, TaskRecord, TaskResultPayload,
    TaskStatus,
};
pub use planning::DistributedRuntimePlanner;
pub use topology::{
    ClusterInventory, DeviceCapabilities, DeviceTopology, HostPlatform, InterconnectKind, PeerLink,
};
pub use worker::{
    InProcessWorker, Worker, WorkerExecutor, WorkerRegistration,
};
#[cfg(feature = "local-executor")]
pub use worker::LocalEncodeWorkerExecutor;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn make_device(
        node_id: &str,
        device_id: usize,
        total_memory_bytes: u64,
        free_memory_bytes: u64,
        max_safe_allocation_bytes: u64,
        measured_encode_samples_per_sec: Option<f64>,
        host_platform: HostPlatform,
        stability_factor: f64,
    ) -> DeviceCapabilities {
        DeviceCapabilities {
            node_id: node_id.to_string(),
            device_id,
            device_name: format!("gpu-{}", device_id),
            total_memory_bytes,
            free_memory_bytes,
            max_safe_allocation_bytes,
            measured_encode_samples_per_sec,
            host_platform,
            stability_factor,
            peer_links: Vec::new(),
        }
    }

    #[derive(Clone, Debug)]
    struct MetricWorker {
        registration: WorkerRegistration,
        metric_name: String,
        value_repr: String,
    }

    impl Worker for MetricWorker {
        fn registration(&self) -> &WorkerRegistration {
            &self.registration
        }
    }

    impl WorkerExecutor for MetricWorker {
        fn execute_task(&self, task: &PartitionTask) -> qdp_core::Result<PartitionTaskResult> {
            Ok(PartitionTaskResult {
                job_id: task.job_id.clone(),
                partition_id: task.partition_id,
                worker_id: self.registration.worker_id.clone(),
                success: true,
                payload: Some(TaskResultPayload::ReducedMetric {
                    metric_name: self.metric_name.clone(),
                    value_repr: self.value_repr.clone(),
                }),
                error: None,
            })
        }
    }

    #[derive(Clone, Debug)]
    struct FlakyWorker {
        registration: WorkerRegistration,
    }

    impl Worker for FlakyWorker {
        fn registration(&self) -> &WorkerRegistration {
            &self.registration
        }
    }

    impl WorkerExecutor for FlakyWorker {
        fn execute_task(&self, task: &PartitionTask) -> qdp_core::Result<PartitionTaskResult> {
            Ok(PartitionTaskResult {
                job_id: task.job_id.clone(),
                partition_id: task.partition_id,
                worker_id: self.registration.worker_id.clone(),
                success: false,
                payload: None,
                error: Some("injected failure".to_string()),
            })
        }
    }

    #[test]
    fn partition_layout_derives_local_qubits_and_offsets() {
        let layout = PartitionLayout::new(6, 4).expect("layout");
        assert_eq!(layout.global_qubits, 6);
        assert_eq!(layout.local_qubits, 4);
        assert_eq!(layout.total_amplitudes, 64);
        assert_eq!(layout.amplitudes_per_partition, 16);
    }

    #[test]
    fn partition_layout_rejects_non_power_of_two_partition_count() {
        let err = PartitionLayout::new(6, 3).unwrap_err();
        assert!(matches!(err, qdp_core::MahoutError::InvalidInput(_)));
    }

    #[test]
    fn weighted_planner_prefers_stronger_device() {
        let devices = vec![
            make_device("node-a", 0, 48, 40, 40, Some(4000.0), HostPlatform::Linux, 1.0),
            make_device("node-b", 0, 24, 12, 12, Some(1000.0), HostPlatform::Wsl, 0.8),
        ];

        let assignments =
            DistributedRuntimePlanner::plan_partitions(8, &devices, PlacementPolicy::Weighted)
                .expect("plan");
        let strong = assignments.iter().filter(|a| a.node_id == "node-a").count();
        let weak = assignments.iter().filter(|a| a.node_id == "node-b").count();
        assert!(strong > weak);
    }

    #[test]
    fn build_state_handle_assigns_contiguous_partitions() {
        let devices = vec![
            make_device("node-a", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0),
            make_device("node-b", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0),
        ];

        let handle = DistributedRuntimePlanner::build_state_handle(
            "state-1",
            5,
            DType::Complex64,
            ConsumptionMode::GatherFullState,
            &devices,
            PlacementPolicy::Weighted,
        )
        .expect("state handle");

        assert_eq!(handle.layout.partition_count, 2);
        assert_eq!(handle.partitions.len(), 2);
        assert_eq!(handle.partitions[0].offset_amplitudes, 0);
        assert_eq!(handle.partitions[1].offset_amplitudes, 16);
        assert_eq!(handle.partitions[0].amplitude_len, 16);
        assert_eq!(handle.partitions[1].amplitude_len, 16);
    }

    #[test]
    fn topology_aware_planner_prefers_nvlink_island() {
        let mut gpu0 = make_device("node-a", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0);
        let mut gpu1 = make_device("node-a", 1, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0);
        gpu0.peer_links.push(PeerLink {
            peer_node_id: "node-a".to_string(),
            peer_device_id: 1,
            kind: InterconnectKind::Nvlink,
            bandwidth_rank: 4,
        });
        gpu1.peer_links.push(PeerLink {
            peer_node_id: "node-a".to_string(),
            peer_device_id: 0,
            kind: InterconnectKind::Nvlink,
            bandwidth_rank: 4,
        });
        let weak = make_device("node-b", 0, 24, 12, 12, Some(1200.0), HostPlatform::Linux, 1.0);

        let assignments = DistributedRuntimePlanner::plan_partitions(
            4,
            &[gpu0, gpu1, weak],
            PlacementPolicy::TopologyAware,
        )
        .expect("topology-aware plan");

        let nvlink_assignments = assignments.iter().filter(|a| a.node_id == "node-a").count();
        assert!(nvlink_assignments >= 3);
    }

    #[test]
    fn coordinator_registers_workers_and_plans_job() {
        let worker_a = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-a".to_string(),
            node_id: "node-a".to_string(),
            devices: vec![make_device("node-a", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0)],
        })
        .expect("worker-a");
        let worker_b = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-b".to_string(),
            node_id: "node-b".to_string(),
            devices: vec![make_device("node-b", 0, 24, 20, 20, Some(2000.0), HostPlatform::Linux, 1.0)],
        })
        .expect("worker-b");

        let mut coordinator = Coordinator::new();
        coordinator.register_worker(&worker_a).expect("register a");
        coordinator.register_worker(&worker_b).expect("register b");
        assert_eq!(coordinator.worker_count(), 2);
        assert_eq!(coordinator.all_devices().len(), 2);

        let planned = coordinator
            .plan_job(RuntimeJobSpec {
                job_id: "job-1".to_string(),
                state_id: "state-1".to_string(),
                global_qubits: 5,
                dtype: DType::Complex64,
                consumption_mode: ConsumptionMode::GatherFullState,
                placement_policy: PlacementPolicy::Weighted,
            })
            .expect("plan job");

        assert_eq!(planned.spec.job_id, "job-1");
        assert_eq!(planned.status, JobStatus::Planned);
        assert_eq!(planned.state.layout.partition_count, 2);
        assert_eq!(planned.tasks.len(), 2);
    }

    #[test]
    fn coordinator_assigns_and_completes_tasks() {
        let worker = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-a".to_string(),
            node_id: "node-a".to_string(),
            devices: vec![make_device("node-a", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0)],
        })
        .expect("worker");

        let mut coordinator = Coordinator::new();
        coordinator.register_worker(&worker).expect("register worker");
        coordinator
            .plan_job(RuntimeJobSpec {
                job_id: "job-2".to_string(),
                state_id: "state-2".to_string(),
                global_qubits: 4,
                dtype: DType::Complex64,
                consumption_mode: ConsumptionMode::GatherFullState,
                placement_policy: PlacementPolicy::Weighted,
            })
            .expect("plan job");

        let task = coordinator
            .assign_next_task("worker-a")
            .expect("assign task")
            .expect("task exists");
        coordinator
            .mark_task_running("job-2", task.partition_id, "worker-a")
            .expect("mark running");
        coordinator
            .report_task_result(PartitionTaskResult {
                job_id: "job-2".to_string(),
                partition_id: task.partition_id,
                worker_id: "worker-a".to_string(),
                success: true,
                payload: Some(TaskResultPayload::PartitionReady {
                    storage_handle: "state:state-2:partition:0".to_string(),
                }),
                error: None,
            })
            .expect("report");

        assert_eq!(
            coordinator.task_record("job-2", task.partition_id).unwrap().status,
            TaskStatus::Completed
        );
        assert_eq!(coordinator.job("job-2").unwrap().status, JobStatus::Completed);
    }

    #[test]
    fn coordinator_run_loop_completes_job_with_in_process_workers() {
        let worker_a = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-a".to_string(),
            node_id: "node-a".to_string(),
            devices: vec![make_device("node-a", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0)],
        })
        .expect("worker-a");
        let worker_b = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-b".to_string(),
            node_id: "node-b".to_string(),
            devices: vec![make_device("node-b", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0)],
        })
        .expect("worker-b");

        let mut coordinator = Coordinator::new();
        coordinator.register_worker(&worker_a).expect("register a");
        coordinator.register_worker(&worker_b).expect("register b");
        coordinator
            .plan_job(RuntimeJobSpec {
                job_id: "job-3".to_string(),
                state_id: "state-3".to_string(),
                global_qubits: 5,
                dtype: DType::Complex64,
                consumption_mode: ConsumptionMode::GatherFullState,
                placement_policy: PlacementPolicy::Weighted,
            })
            .expect("plan job");

        let completed = coordinator
            .run_job_with_workers("job-3", &[worker_a, worker_b])
            .expect("run");
        assert_eq!(completed.status, JobStatus::Completed);
    }

    #[test]
    fn coordinator_builds_gather_plan_for_completed_job() {
        let worker_a = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-a".to_string(),
            node_id: "node-a".to_string(),
            devices: vec![make_device("node-a", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0)],
        })
        .expect("worker-a");
        let worker_b = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-b".to_string(),
            node_id: "node-b".to_string(),
            devices: vec![make_device("node-b", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0)],
        })
        .expect("worker-b");

        let mut coordinator = Coordinator::new();
        coordinator.register_worker(&worker_a).expect("register a");
        coordinator.register_worker(&worker_b).expect("register b");
        coordinator
            .plan_job(RuntimeJobSpec {
                job_id: "job-4".to_string(),
                state_id: "state-4".to_string(),
                global_qubits: 5,
                dtype: DType::Complex64,
                consumption_mode: ConsumptionMode::GatherFullState,
                placement_policy: PlacementPolicy::Weighted,
            })
            .expect("plan");
        coordinator.run_job_with_workers("job-4", &[worker_a, worker_b]).expect("run");

        let gather_plan = coordinator
            .build_gather_plan("job-4", GatherTarget::HostMemory)
            .expect("gather");
        assert_eq!(coordinator.objects_for_job("job-4").len(), 2);
        assert!(coordinator.object("obj:job-4:partition:0").is_some());
        assert_eq!(gather_plan.total_amplitudes, 32);
    }

    #[test]
    fn coordinator_builds_reduce_plan_for_completed_job() {
        let worker_a = MetricWorker {
            registration: WorkerRegistration {
                worker_id: "worker-a".to_string(),
                node_id: "node-a".to_string(),
                devices: vec![make_device("node-a", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0)],
            },
            metric_name: "loss".to_string(),
            value_repr: "1.25".to_string(),
        };
        let worker_b = MetricWorker {
            registration: WorkerRegistration {
                worker_id: "worker-b".to_string(),
                node_id: "node-b".to_string(),
                devices: vec![make_device("node-b", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0)],
            },
            metric_name: "loss".to_string(),
            value_repr: "2.50".to_string(),
        };

        let mut coordinator = Coordinator::new();
        coordinator.register_worker(&worker_a).expect("register a");
        coordinator.register_worker(&worker_b).expect("register b");
        coordinator
            .plan_job(RuntimeJobSpec {
                job_id: "job-5".to_string(),
                state_id: "state-5".to_string(),
                global_qubits: 5,
                dtype: DType::Complex64,
                consumption_mode: ConsumptionMode::ReduceMetrics,
                placement_policy: PlacementPolicy::Weighted,
            })
            .expect("plan");
        coordinator.run_job_with_workers("job-5", &[worker_a, worker_b]).expect("run");

        let reduce_plan = coordinator
            .build_reduce_plan("job-5", "loss", MetricReduceOp::Mean)
            .expect("reduce plan");
        let reduced = coordinator.execute_reduce_plan(&reduce_plan).expect("reduce");
        assert_eq!(reduced, ReducedMetricValue::Scalar(1.875));
    }

    #[test]
    fn failed_task_requeues_before_max_attempts() {
        let worker = FlakyWorker {
            registration: WorkerRegistration {
                worker_id: "worker-a".to_string(),
                node_id: "node-a".to_string(),
                devices: vec![make_device("node-a", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0)],
            },
        };

        let mut coordinator = Coordinator::with_retry_policy(RetryPolicy {
            max_attempts: 2,
            lease_timeout: Duration::from_secs(30),
        });
        coordinator.register_worker(&worker).expect("register");
        coordinator
            .plan_job(RuntimeJobSpec {
                job_id: "job-6".to_string(),
                state_id: "state-6".to_string(),
                global_qubits: 4,
                dtype: DType::Complex64,
                consumption_mode: ConsumptionMode::GatherFullState,
                placement_policy: PlacementPolicy::Weighted,
            })
            .expect("plan");

        let task = coordinator.assign_next_task("worker-a").unwrap().unwrap();
        coordinator.mark_task_running("job-6", task.partition_id, "worker-a").unwrap();
        let result = worker.execute_task(&task).unwrap();
        coordinator.report_task_result(result).unwrap();

        let record = coordinator.task_record("job-6", task.partition_id).unwrap();
        assert_eq!(record.status, TaskStatus::Pending);
    }

    #[test]
    fn inventory_and_topology_resolve_nvlink_peers() {
        let mut device_a0 = make_device("node-a", 0, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0);
        let mut device_a1 = make_device("node-a", 1, 48, 40, 40, Some(3000.0), HostPlatform::Linux, 1.0);
        device_a0.peer_links.push(PeerLink {
            peer_node_id: "node-a".to_string(),
            peer_device_id: 1,
            kind: InterconnectKind::Nvlink,
            bandwidth_rank: 4,
        });
        device_a1.peer_links.push(PeerLink {
            peer_node_id: "node-a".to_string(),
            peer_device_id: 0,
            kind: InterconnectKind::Nvlink,
            bandwidth_rank: 4,
        });

        let inventory = ClusterInventory::from_workers(vec![
            WorkerRegistration {
                worker_id: "worker-a".to_string(),
                node_id: "node-a".to_string(),
                devices: vec![device_a0.clone(), device_a1.clone()],
            },
            WorkerRegistration {
                worker_id: "worker-b".to_string(),
                node_id: "node-b".to_string(),
                devices: vec![make_device("node-b", 0, 24, 12, 12, Some(1000.0), HostPlatform::Linux, 1.0)],
            },
        ]);

        assert_eq!(inventory.devices().len(), 3);
        assert!(inventory.worker_for_device("node-a", 1).is_some());
        let topology = DeviceTopology::from_inventory(&inventory);
        assert_eq!(topology.nvlink_peers("node-a", 0).len(), 1);
        assert!(topology.same_nvlink_island("node-a", 0, "node-a", 1));
        assert!(!topology.same_nvlink_island("node-a", 0, "node-b", 0));
    }
}
