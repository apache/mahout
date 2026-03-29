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

use std::collections::BTreeMap;
use std::time::Instant;

use qdp_core::{MahoutError, Result};

use crate::model::{
    ConsumptionMode, GatherPlan, GatherSegment, GatherTarget, JobStatus, MetricContribution,
    MetricReduceOp, PartitionTask, PartitionTaskResult, PlannedJob, ReducePlan, ReducedMetricValue,
    RetryPolicy, RuntimeJobSpec, RuntimeObjectKind, RuntimeObjectLocation, RuntimeObjectRecord,
    TaskRecord, TaskResultPayload, TaskStatus,
};
use crate::planning::DistributedRuntimePlanner;
use crate::runtime_profile_scope;
use crate::topology::{ClusterInventory, DeviceTopology};
use crate::worker::{Worker, WorkerExecutor, WorkerRegistration};

pub struct Coordinator {
    workers: BTreeMap<String, WorkerRegistration>,
    jobs: BTreeMap<String, PlannedJob>,
    task_records: BTreeMap<(String, usize), TaskRecord>,
    objects: BTreeMap<String, RuntimeObjectRecord>,
    retry_policy: RetryPolicy,
}

impl Default for Coordinator {
    fn default() -> Self {
        Self::with_retry_policy(RetryPolicy::default())
    }
}

impl Coordinator {
    pub fn new() -> Self {
        Self::with_retry_policy(RetryPolicy::default())
    }

    pub fn with_retry_policy(retry_policy: RetryPolicy) -> Self {
        Self {
            workers: BTreeMap::new(),
            jobs: BTreeMap::new(),
            task_records: BTreeMap::new(),
            objects: BTreeMap::new(),
            retry_policy,
        }
    }

    pub fn register_worker<W: Worker>(&mut self, worker: &W) -> Result<()> {
        runtime_profile_scope!("Coordinator::RegisterWorker");
        let registration = worker.registration().clone();
        registration.validate()?;
        self.workers
            .insert(registration.worker_id.clone(), registration);
        Ok(())
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn cluster_inventory(&self) -> ClusterInventory {
        ClusterInventory::from_workers(self.workers.values().cloned())
    }

    pub fn device_topology(&self) -> DeviceTopology {
        DeviceTopology::from_inventory(&self.cluster_inventory())
    }

    pub fn all_devices(&self) -> Vec<crate::topology::DeviceCapabilities> {
        self.cluster_inventory().devices
    }

    pub fn plan_job(&mut self, spec: RuntimeJobSpec) -> Result<&PlannedJob> {
        runtime_profile_scope!("Coordinator::PlanJob");

        if spec.job_id.is_empty() {
            return Err(MahoutError::InvalidInput("job_id must not be empty".to_string()));
        }
        if self.jobs.contains_key(&spec.job_id) {
            return Err(MahoutError::InvalidInput(format!("job {} already exists", spec.job_id)));
        }

        let devices = self.all_devices();
        if devices.is_empty() {
            return Err(MahoutError::InvalidInput(
                "cannot plan job without registered devices".to_string(),
            ));
        }

        let state = DistributedRuntimePlanner::build_state_handle(
            spec.state_id.clone(),
            spec.global_qubits,
            spec.dtype.clone(),
            spec.consumption_mode.clone(),
            &devices,
            spec.placement_policy,
        )?;
        let tasks = self.build_partition_tasks(&spec, &state)?;

        self.jobs.insert(
            spec.job_id.clone(),
            PlannedJob {
                spec: spec.clone(),
                status: JobStatus::Planned,
                state,
                tasks,
            },
        );

        if let Some(job) = self.jobs.get(&spec.job_id) {
            for task in &job.tasks {
                self.task_records.insert(
                    (task.job_id.clone(), task.partition_id),
                    TaskRecord {
                        task: task.clone(),
                        status: TaskStatus::Pending,
                        attempt_id: 0,
                        last_error: None,
                        last_payload: None,
                        lease_deadline: None,
                    },
                );
            }
        }
        self.refresh_job_status(&spec.job_id)?;
        self.jobs.get(&spec.job_id).ok_or_else(|| {
            MahoutError::InvalidInput(format!("planned job {} was not stored", spec.job_id))
        })
    }

    pub fn job(&self, job_id: &str) -> Option<&PlannedJob> {
        self.jobs.get(job_id)
    }

    pub fn task_record(&self, job_id: &str, partition_id: usize) -> Option<&TaskRecord> {
        self.task_records.get(&(job_id.to_string(), partition_id))
    }

    pub fn object(&self, object_id: &str) -> Option<&RuntimeObjectRecord> {
        self.objects.get(object_id)
    }

    pub fn objects_for_job(&self, job_id: &str) -> Vec<&RuntimeObjectRecord> {
        self.objects
            .values()
            .filter(|object| object.job_id == job_id)
            .collect()
    }

    pub fn assign_next_task(&mut self, worker_id: &str) -> Result<Option<PartitionTask>> {
        runtime_profile_scope!("Coordinator::AssignNextTask");
        self.reap_expired_leases();

        let worker = self.workers.get(worker_id).ok_or_else(|| {
            MahoutError::InvalidInput(format!("worker {} is not registered", worker_id))
        })?;

        let next_key = self.task_records.iter().find_map(|(key, record)| {
            if record.status == TaskStatus::Pending
                && record.task.worker_id == worker.worker_id
                && record.task.node_id == worker.node_id
            {
                Some(key.clone())
            } else {
                None
            }
        });

        let Some(key) = next_key else {
            return Ok(None);
        };

        let record = self.task_records.get_mut(&key).ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "task record for job {} partition {} disappeared",
                key.0, key.1
            ))
        })?;

        if record.attempt_id >= self.retry_policy.max_attempts {
            record.status = TaskStatus::Failed;
            record.last_error = Some(format!(
                "task exceeded max attempts ({})",
                self.retry_policy.max_attempts
            ));
            self.refresh_job_status(&key.0)?;
            return Ok(None);
        }

        record.status = TaskStatus::Assigned;
        record.attempt_id += 1;
        record.lease_deadline = Some(Instant::now() + self.retry_policy.lease_timeout);
        self.refresh_job_status(&key.0)?;
        Ok(Some(record.task.clone()))
    }

    pub fn mark_task_running(
        &mut self,
        job_id: &str,
        partition_id: usize,
        worker_id: &str,
    ) -> Result<()> {
        runtime_profile_scope!("Coordinator::MarkTaskRunning");

        let key = (job_id.to_string(), partition_id);
        let record = self.task_records.get_mut(&key).ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "unknown task for job {} partition {}",
                job_id, partition_id
            ))
        })?;

        if record.task.worker_id != worker_id {
            return Err(MahoutError::InvalidInput(format!(
                "task job {} partition {} is assigned to worker {}, not {}",
                job_id, partition_id, record.task.worker_id, worker_id
            )));
        }

        record.status = TaskStatus::Running;
        record.lease_deadline = Some(Instant::now() + self.retry_policy.lease_timeout);
        self.refresh_job_status(job_id)?;
        Ok(())
    }

    pub fn report_task_result(&mut self, result: PartitionTaskResult) -> Result<()> {
        runtime_profile_scope!("Coordinator::ReportTaskResult");

        let key = (result.job_id.clone(), result.partition_id);
        let record = self.task_records.get_mut(&key).ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "unknown task result for job {} partition {}",
                result.job_id, result.partition_id
            ))
        })?;

        if record.task.worker_id != result.worker_id {
            return Err(MahoutError::InvalidInput(format!(
                "task result worker mismatch for job {} partition {}: expected {}, got {}",
                result.job_id, result.partition_id, record.task.worker_id, result.worker_id
            )));
        }

        if result.success {
            record.status = TaskStatus::Completed;
            record.last_error = None;
            record.last_payload = result.payload.clone();
            record.lease_deadline = None;
        } else {
            let error = result.error.unwrap_or_else(|| "task failed".to_string());
            record.status = if record.attempt_id < self.retry_policy.max_attempts {
                TaskStatus::Pending
            } else {
                TaskStatus::Failed
            };
            record.last_error = Some(error);
            record.last_payload = None;
            record.lease_deadline = None;
        }

        if result.success {
            match result.payload {
                Some(TaskResultPayload::PartitionReady { storage_handle }) => {
                    let object_id =
                        format!("obj:{}:partition:{}", result.job_id, result.partition_id);
                    self.objects.insert(
                        object_id.clone(),
                        RuntimeObjectRecord {
                            object_id,
                            job_id: result.job_id.clone(),
                            partition_id: result.partition_id,
                            kind: RuntimeObjectKind::EncodedPartition,
                            location: RuntimeObjectLocation::Device {
                                node_id: record.task.node_id.clone(),
                                device_id: record.task.device_id,
                            },
                            handle: storage_handle,
                            ready: true,
                        },
                    );
                }
                Some(TaskResultPayload::ReducedMetric {
                    metric_name,
                    value_repr,
                }) => {
                    let object_id = format!(
                        "obj:{}:partition:{}:metric:{}",
                        result.job_id, result.partition_id, metric_name
                    );
                    self.objects.insert(
                        object_id.clone(),
                        RuntimeObjectRecord {
                            object_id,
                            job_id: result.job_id.clone(),
                            partition_id: result.partition_id,
                            kind: RuntimeObjectKind::ReducedMetric,
                            location: RuntimeObjectLocation::Host,
                            handle: format!("metric:{}={}", metric_name, value_repr),
                            ready: true,
                        },
                    );
                }
                None => {}
            }
        }

        self.refresh_job_status(&result.job_id)?;
        Ok(())
    }

    pub fn build_gather_plan(&self, job_id: &str, target: GatherTarget) -> Result<GatherPlan> {
        runtime_profile_scope!("Coordinator::BuildGatherPlan");

        let job = self
            .jobs
            .get(job_id)
            .ok_or_else(|| MahoutError::InvalidInput(format!("unknown job {}", job_id)))?;
        if job.spec.consumption_mode != ConsumptionMode::GatherFullState {
            return Err(MahoutError::InvalidInput(format!(
                "job {} does not use GatherFullState consumption mode",
                job_id
            )));
        }
        if job.status != JobStatus::Completed {
            return Err(MahoutError::InvalidInput(format!(
                "job {} must be completed before building a gather plan",
                job_id
            )));
        }

        let mut segments = Vec::with_capacity(job.state.partitions.len());
        for partition in &job.state.partitions {
            let object_id = format!("obj:{}:partition:{}", job_id, partition.partition_id);
            let object = self.objects.get(&object_id).ok_or_else(|| {
                MahoutError::InvalidInput(format!(
                    "missing runtime object {} for job {} partition {}",
                    object_id, job_id, partition.partition_id
                ))
            })?;

            if object.kind != RuntimeObjectKind::EncodedPartition || !object.ready {
                return Err(MahoutError::InvalidInput(format!(
                    "runtime object {} is not a ready encoded partition",
                    object_id
                )));
            }

            segments.push(GatherSegment {
                partition_id: partition.partition_id,
                source_node_id: partition.node_id.clone(),
                source_device_id: partition.device_id,
                source_storage_handle: object.handle.clone(),
                destination_offset_amplitudes: partition.offset_amplitudes,
                amplitude_len: partition.amplitude_len,
            });
        }
        segments.sort_by_key(|segment| segment.destination_offset_amplitudes);

        Ok(GatherPlan {
            job_id: job.spec.job_id.clone(),
            state_id: job.state.state_id.clone(),
            target,
            total_amplitudes: job.state.layout.total_amplitudes,
            segments,
        })
    }

    pub fn build_reduce_plan(
        &self,
        job_id: &str,
        metric_name: &str,
        op: MetricReduceOp,
    ) -> Result<ReducePlan> {
        runtime_profile_scope!("Coordinator::BuildReducePlan");

        let job = self
            .jobs
            .get(job_id)
            .ok_or_else(|| MahoutError::InvalidInput(format!("unknown job {}", job_id)))?;
        if job.spec.consumption_mode != ConsumptionMode::ReduceMetrics {
            return Err(MahoutError::InvalidInput(format!(
                "job {} does not use ReduceMetrics consumption mode",
                job_id
            )));
        }
        if job.status != JobStatus::Completed {
            return Err(MahoutError::InvalidInput(format!(
                "job {} must be completed before building a reduce plan",
                job_id
            )));
        }
        if metric_name.is_empty() {
            return Err(MahoutError::InvalidInput(
                "metric_name must not be empty".to_string(),
            ));
        }

        let mut contributions = Vec::new();
        for ((task_job_id, partition_id), record) in &self.task_records {
            if task_job_id != job_id {
                continue;
            }
            let Some(TaskResultPayload::ReducedMetric {
                metric_name: payload_metric_name,
                value_repr,
            }) = &record.last_payload
            else {
                continue;
            };
            if payload_metric_name != metric_name {
                continue;
            }
            contributions.push(MetricContribution {
                partition_id: *partition_id,
                worker_id: record.task.worker_id.clone(),
                metric_name: payload_metric_name.clone(),
                value_repr: value_repr.clone(),
            });
        }

        if contributions.is_empty() {
            return Err(MahoutError::InvalidInput(format!(
                "no metric contributions found for job {} metric {}",
                job_id, metric_name
            )));
        }
        contributions.sort_by_key(|contribution| contribution.partition_id);

        Ok(ReducePlan {
            job_id: job.spec.job_id.clone(),
            metric_name: metric_name.to_string(),
            op,
            contributions,
        })
    }

    pub fn execute_reduce_plan(&self, plan: &ReducePlan) -> Result<ReducedMetricValue> {
        runtime_profile_scope!("Coordinator::ExecuteReducePlan");

        if plan.contributions.is_empty() {
            return Err(MahoutError::InvalidInput(format!(
                "reduce plan for job {} metric {} has no contributions",
                plan.job_id, plan.metric_name
            )));
        }

        match plan.op {
            MetricReduceOp::Concat => {
                let joined = plan
                    .contributions
                    .iter()
                    .map(|contribution| contribution.value_repr.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                Ok(ReducedMetricValue::Text(joined))
            }
            MetricReduceOp::Sum
            | MetricReduceOp::Mean
            | MetricReduceOp::Min
            | MetricReduceOp::Max => {
                let values = plan
                    .contributions
                    .iter()
                    .map(|contribution| {
                        contribution.value_repr.parse::<f64>().map_err(|_| {
                            MahoutError::InvalidInput(format!(
                                "failed to parse metric contribution '{}' as f64 for metric {}",
                                contribution.value_repr, plan.metric_name
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                let reduced = match plan.op {
                    MetricReduceOp::Sum => values.iter().sum(),
                    MetricReduceOp::Mean => values.iter().sum::<f64>() / values.len() as f64,
                    MetricReduceOp::Min => values.iter().copied().reduce(f64::min).unwrap(),
                    MetricReduceOp::Max => values.iter().copied().reduce(f64::max).unwrap(),
                    MetricReduceOp::Concat => unreachable!(),
                };
                Ok(ReducedMetricValue::Scalar(reduced))
            }
        }
    }

    pub fn run_job_with_workers<W>(&mut self, job_id: &str, workers: &[W]) -> Result<&PlannedJob>
    where
        W: WorkerExecutor,
    {
        runtime_profile_scope!("Coordinator::RunJobWithWorkers");

        if !self.jobs.contains_key(job_id) {
            return Err(MahoutError::InvalidInput(format!("unknown job {}", job_id)));
        }
        if workers.is_empty() {
            return Err(MahoutError::InvalidInput(
                "at least one worker is required to run a job".to_string(),
            ));
        }

        loop {
            let status = self
                .job(job_id)
                .ok_or_else(|| MahoutError::InvalidInput(format!("unknown job {}", job_id)))?
                .status
                .clone();
            match status {
                JobStatus::Completed | JobStatus::Failed => break,
                JobStatus::Pending | JobStatus::Planned | JobStatus::Running => {}
            }

            let mut made_progress = false;
            for worker in workers {
                let worker_id = worker.registration().worker_id.clone();
                let Some(task) = self.assign_next_task(&worker_id)? else {
                    continue;
                };
                self.mark_task_running(&task.job_id, task.partition_id, &worker_id)?;
                let result = worker.execute_task(&task)?;
                self.report_task_result(result)?;
                made_progress = true;
            }

            if !made_progress {
                let status = self
                    .job(job_id)
                    .ok_or_else(|| MahoutError::InvalidInput(format!("unknown job {}", job_id)))?
                    .status
                    .clone();
                if status != JobStatus::Completed && status != JobStatus::Failed {
                    return Err(MahoutError::InvalidInput(format!(
                        "job {} made no progress; all matching workers may be idle or missing",
                        job_id
                    )));
                }
            }
        }

        self.job(job_id)
            .ok_or_else(|| MahoutError::InvalidInput(format!("unknown job {}", job_id)))
    }

    fn build_partition_tasks(
        &self,
        spec: &RuntimeJobSpec,
        state: &crate::model::DistributedStateHandle,
    ) -> Result<Vec<PartitionTask>> {
        runtime_profile_scope!("Coordinator::BuildPartitionTasks");

        let inventory = self.cluster_inventory();
        let mut worker_by_device = BTreeMap::<(String, usize), String>::new();
        for worker in &inventory.workers {
            for device in &worker.devices {
                worker_by_device
                    .insert((device.node_id.clone(), device.device_id), worker.worker_id.clone());
            }
        }

        state
            .partitions
            .iter()
            .map(|partition| {
                let worker_id = worker_by_device
                    .get(&(partition.node_id.clone(), partition.device_id))
                    .cloned()
                    .ok_or_else(|| {
                        MahoutError::InvalidInput(format!(
                            "no worker registered for node {} device {}",
                            partition.node_id, partition.device_id
                        ))
                    })?;

                Ok(PartitionTask {
                    job_id: spec.job_id.clone(),
                    state_id: partition.state_id.clone(),
                    partition_id: partition.partition_id,
                    worker_id,
                    node_id: partition.node_id.clone(),
                    device_id: partition.device_id,
                    offset_amplitudes: partition.offset_amplitudes,
                    amplitude_len: partition.amplitude_len,
                    global_qubits: partition.global_qubits,
                    local_qubits: partition.local_qubits,
                    consumption_mode: spec.consumption_mode.clone(),
                })
            })
            .collect()
    }

    fn reap_expired_leases(&mut self) {
        let now = Instant::now();
        for record in self.task_records.values_mut() {
            if matches!(record.status, TaskStatus::Assigned | TaskStatus::Running)
                && record.lease_deadline.is_some_and(|deadline| deadline <= now)
            {
                if record.attempt_id < self.retry_policy.max_attempts {
                    record.status = TaskStatus::Pending;
                    record.last_error = Some("task lease expired".to_string());
                } else {
                    record.status = TaskStatus::Failed;
                    record.last_error =
                        Some(format!("task lease expired after {} attempts", record.attempt_id));
                }
                record.last_payload = None;
                record.lease_deadline = None;
            }
        }
    }

    fn refresh_job_status(&mut self, job_id: &str) -> Result<()> {
        let mut task_count = 0usize;
        let mut has_failed = false;
        let mut all_completed = true;
        let mut has_active = false;

        for ((task_job_id, _), record) in &self.task_records {
            if task_job_id != job_id {
                continue;
            }
            task_count += 1;
            match record.status {
                TaskStatus::Completed => {}
                TaskStatus::Failed => {
                    has_failed = true;
                    all_completed = false;
                }
                TaskStatus::Pending => {
                    all_completed = false;
                }
                TaskStatus::Assigned | TaskStatus::Running => {
                    has_active = true;
                    all_completed = false;
                }
            }
        }

        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| MahoutError::InvalidInput(format!("unknown job {}", job_id)))?;

        job.status = if task_count == 0 {
            JobStatus::Planned
        } else if has_failed {
            JobStatus::Failed
        } else if all_completed {
            JobStatus::Completed
        } else if has_active {
            JobStatus::Running
        } else {
            JobStatus::Planned
        };
        Ok(())
    }
}
