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

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[cfg(feature = "local-executor")]
use qdp_core::dlpack::free_dlpack_tensor;
use qdp_core::{MahoutError, Result};
#[cfg(feature = "local-executor")]
use qdp_core::{Precision, QdpEngine};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartitionScheme {
    ContiguousAmplitudeBlocks,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DType {
    Complex64,
    Complex128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConsumptionMode {
    PartitionLocalConsume,
    GatherFullState,
    ReduceMetrics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementPolicy {
    RoundRobin,
    Weighted,
    TopologyAware,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostPlatform {
    Linux,
    Wsl,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterconnectKind {
    Nvlink,
    Pcie,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerLink {
    pub peer_node_id: String,
    pub peer_device_id: usize,
    pub kind: InterconnectKind,
    /// Relative bandwidth hint; larger values indicate faster peer-to-peer movement.
    pub bandwidth_rank: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceCapabilities {
    pub node_id: String,
    pub device_id: usize,
    pub device_name: String,
    pub total_memory_bytes: u64,
    pub free_memory_bytes: u64,
    pub max_safe_allocation_bytes: u64,
    pub measured_encode_samples_per_sec: Option<f64>,
    pub host_platform: HostPlatform,
    /// Multiplier in (0, 1]; lower values penalize weaker or less stable workers.
    pub stability_factor: f64,
    /// Topology hints for peer-aware placement. For v1 this is advisory metadata only.
    pub peer_links: Vec<PeerLink>,
}

impl DeviceCapabilities {
    fn placement_weight(&self) -> Result<f64> {
        if self.total_memory_bytes == 0 {
            return Err(MahoutError::InvalidInput(format!(
                "Device {} on node {} has zero total memory",
                self.device_id, self.node_id
            )));
        }
        if !(0.0 < self.stability_factor && self.stability_factor <= 1.0) {
            return Err(MahoutError::InvalidInput(format!(
                "Device {} on node {} has invalid stability_factor {}",
                self.device_id, self.node_id, self.stability_factor
            )));
        }

        let memory_factor = self.max_safe_allocation_bytes.min(self.free_memory_bytes) as f64
            / self.total_memory_bytes as f64;
        let throughput_factor = self.measured_encode_samples_per_sec.unwrap_or(1.0).max(1.0);
        let host_penalty = match self.host_platform {
            HostPlatform::Linux => 1.0,
            HostPlatform::Wsl => 0.8,
            HostPlatform::Other => 0.7,
        };

        Ok(memory_factor.max(0.05) * throughput_factor * host_penalty * self.stability_factor)
    }

    fn nvlink_peer_count(&self) -> usize {
        self.peer_links
            .iter()
            .filter(|link| link.kind == InterconnectKind::Nvlink)
            .count()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionLayout {
    pub global_qubits: u32,
    pub local_qubits: u32,
    pub partition_count: usize,
    pub total_amplitudes: usize,
    pub amplitudes_per_partition: usize,
    pub scheme: PartitionScheme,
}

impl PartitionLayout {
    pub fn new(global_qubits: u32, partition_count: usize) -> Result<Self> {
        runtime_profile_scope!("Runtime::PartitionLayout");

        if partition_count == 0 {
            return Err(MahoutError::InvalidInput(
                "partition_count must be greater than zero".to_string(),
            ));
        }
        if !partition_count.is_power_of_two() {
            return Err(MahoutError::InvalidInput(format!(
                "partition_count must be a power of two, got {}",
                partition_count
            )));
        }
        let partition_bits = partition_count.ilog2();
        if partition_bits > global_qubits {
            return Err(MahoutError::InvalidInput(format!(
                "partition_count={} exceeds state capacity for global_qubits={}",
                partition_count, global_qubits
            )));
        }

        let total_amplitudes = 1usize
            .checked_shl(global_qubits)
            .ok_or_else(|| MahoutError::InvalidInput(format!("global_qubits={} too large", global_qubits)))?;
        let amplitudes_per_partition = total_amplitudes / partition_count;
        let local_qubits = global_qubits - partition_bits;

        Ok(Self {
            global_qubits,
            local_qubits,
            partition_count,
            total_amplitudes,
            amplitudes_per_partition,
            scheme: PartitionScheme::ContiguousAmplitudeBlocks,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatePartitionRef {
    pub state_id: String,
    pub partition_id: usize,
    pub node_id: String,
    pub device_id: usize,
    pub offset_amplitudes: usize,
    pub amplitude_len: usize,
    pub global_qubits: u32,
    pub local_qubits: u32,
    pub storage_handle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedStateHandle {
    pub state_id: String,
    pub dtype: DType,
    pub scheme: PartitionScheme,
    pub consumption_mode: ConsumptionMode,
    pub layout: PartitionLayout,
    pub partitions: Vec<StatePartitionRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GatherTarget {
    HostMemory,
    NodeMemory { node_id: String },
    DeviceMemory { node_id: String, device_id: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherSegment {
    pub partition_id: usize,
    pub source_node_id: String,
    pub source_device_id: usize,
    pub source_storage_handle: String,
    pub destination_offset_amplitudes: usize,
    pub amplitude_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherPlan {
    pub job_id: String,
    pub state_id: String,
    pub target: GatherTarget,
    pub total_amplitudes: usize,
    pub segments: Vec<GatherSegment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetricReduceOp {
    Sum,
    Mean,
    Min,
    Max,
    Concat,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricContribution {
    pub partition_id: usize,
    pub worker_id: String,
    pub metric_name: String,
    pub value_repr: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducePlan {
    pub job_id: String,
    pub metric_name: String,
    pub op: MetricReduceOp,
    pub contributions: Vec<MetricContribution>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReducedMetricValue {
    Scalar(f64),
    Text(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeObjectKind {
    EncodedPartition,
    ReducedMetric,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeObjectLocation {
    Device { node_id: String, device_id: usize },
    Host,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeObjectRecord {
    pub object_id: String,
    pub job_id: String,
    pub partition_id: usize,
    pub kind: RuntimeObjectKind,
    pub location: RuntimeObjectLocation,
    pub handle: String,
    pub ready: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterInventory {
    pub workers: Vec<WorkerRegistration>,
    pub devices: Vec<DeviceCapabilities>,
}

impl ClusterInventory {
    pub fn from_workers(workers: impl IntoIterator<Item = WorkerRegistration>) -> Self {
        let workers = workers.into_iter().collect::<Vec<_>>();
        let devices = workers
            .iter()
            .flat_map(|worker| worker.devices.iter().cloned())
            .collect();
        Self { workers, devices }
    }

    pub fn devices(&self) -> &[DeviceCapabilities] {
        &self.devices
    }

    pub fn worker_for_device(&self, node_id: &str, device_id: usize) -> Option<&WorkerRegistration> {
        self.workers.iter().find(|worker| {
            worker.node_id == node_id
                && worker
                    .devices
                    .iter()
                    .any(|device| device.node_id == node_id && device.device_id == device_id)
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceTopology {
    pub devices: Vec<DeviceCapabilities>,
}

impl DeviceTopology {
    pub fn from_inventory(inventory: &ClusterInventory) -> Self {
        Self {
            devices: inventory.devices.clone(),
        }
    }

    pub fn nvlink_peers(&self, node_id: &str, device_id: usize) -> Vec<&PeerLink> {
        self.devices
            .iter()
            .find(|device| device.node_id == node_id && device.device_id == device_id)
            .map(|device| {
                device
                    .peer_links
                    .iter()
                    .filter(|link| link.kind == InterconnectKind::Nvlink)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn same_nvlink_island(
        &self,
        left_node_id: &str,
        left_device_id: usize,
        right_node_id: &str,
        right_device_id: usize,
    ) -> bool {
        self.nvlink_peers(left_node_id, left_device_id)
            .into_iter()
            .any(|peer| peer.peer_node_id == right_node_id && peer.peer_device_id == right_device_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeJobSpec {
    pub job_id: String,
    pub state_id: String,
    pub global_qubits: u32,
    pub dtype: DType,
    pub consumption_mode: ConsumptionMode,
    pub placement_policy: PlacementPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Planned,
    Running,
    Completed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedJob {
    pub spec: RuntimeJobSpec,
    pub status: JobStatus,
    pub state: DistributedStateHandle,
    pub tasks: Vec<PartitionTask>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionAssignment {
    pub partition_id: usize,
    pub node_id: String,
    pub device_id: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkerRegistration {
    pub worker_id: String,
    pub node_id: String,
    pub devices: Vec<DeviceCapabilities>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionTask {
    pub job_id: String,
    pub state_id: String,
    pub partition_id: usize,
    pub worker_id: String,
    pub node_id: String,
    pub device_id: usize,
    pub offset_amplitudes: usize,
    pub amplitude_len: usize,
    pub global_qubits: u32,
    pub local_qubits: u32,
    pub consumption_mode: ConsumptionMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Assigned,
    Running,
    Completed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskRecord {
    pub task: PartitionTask,
    pub status: TaskStatus,
    pub attempt_id: u32,
    pub last_error: Option<String>,
    pub last_payload: Option<TaskResultPayload>,
    pub lease_deadline: Option<Instant>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskResultPayload {
    PartitionReady { storage_handle: String },
    ReducedMetric { metric_name: String, value_repr: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionTaskResult {
    pub job_id: String,
    pub partition_id: usize,
    pub worker_id: String,
    pub success: bool,
    pub payload: Option<TaskResultPayload>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PartitionInput {
    pub num_qubits: usize,
    pub encoding_method: String,
    pub data: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub lease_timeout: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            lease_timeout: Duration::from_secs(30),
        }
    }
}

impl WorkerRegistration {
    pub fn validate(&self) -> Result<()> {
        if self.worker_id.is_empty() {
            return Err(MahoutError::InvalidInput(
                "worker_id must not be empty".to_string(),
            ));
        }
        if self.node_id.is_empty() {
            return Err(MahoutError::InvalidInput(
                "node_id must not be empty".to_string(),
            ));
        }
        if self.devices.is_empty() {
            return Err(MahoutError::InvalidInput(format!(
                "worker {} has no registered devices",
                self.worker_id
            )));
        }
        for device in &self.devices {
            if device.node_id != self.node_id {
                return Err(MahoutError::InvalidInput(format!(
                    "worker {} node_id={} does not match device node_id={} for device {}",
                    self.worker_id, self.node_id, device.node_id, device.device_id
                )));
            }
        }
        Ok(())
    }
}

pub trait Worker {
    fn registration(&self) -> &WorkerRegistration;
}

pub trait WorkerExecutor: Worker {
    fn execute_task(&self, task: &PartitionTask) -> Result<PartitionTaskResult>;
}

#[derive(Clone, Debug)]
pub struct InProcessWorker {
    registration: WorkerRegistration,
}

impl InProcessWorker {
    pub fn new(registration: WorkerRegistration) -> Result<Self> {
        registration.validate()?;
        Ok(Self { registration })
    }
}

impl Worker for InProcessWorker {
    fn registration(&self) -> &WorkerRegistration {
        &self.registration
    }
}

impl WorkerExecutor for InProcessWorker {
    fn execute_task(&self, task: &PartitionTask) -> Result<PartitionTaskResult> {
        runtime_profile_scope!("Worker::ExecuteTask");

        if task.worker_id != self.registration.worker_id {
            return Err(MahoutError::InvalidInput(format!(
                "task for worker {} was sent to worker {}",
                task.worker_id, self.registration.worker_id
            )));
        }

        Ok(PartitionTaskResult {
            job_id: task.job_id.clone(),
            partition_id: task.partition_id,
            worker_id: self.registration.worker_id.clone(),
            success: true,
            payload: Some(TaskResultPayload::PartitionReady {
                storage_handle: format!(
                    "state:{}:partition:{}:worker:{}",
                    task.state_id, task.partition_id, self.registration.worker_id
                ),
            }),
            error: None,
        })
    }
}

#[cfg(feature = "local-executor")]
#[derive(Clone, Debug)]
pub struct LocalEncodeWorkerExecutor {
    registration: WorkerRegistration,
    engine: QdpEngine,
    partition_inputs: BTreeMap<(String, usize), PartitionInput>,
}

#[cfg(feature = "local-executor")]
impl LocalEncodeWorkerExecutor {
    pub fn new(
        registration: WorkerRegistration,
        precision: Precision,
        partition_inputs: BTreeMap<(String, usize), PartitionInput>,
    ) -> Result<Self> {
        registration.validate()?;
        let primary_device = registration.devices.first().ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "worker {} has no devices for local executor",
                registration.worker_id
            ))
        })?;
        let engine = QdpEngine::new_with_precision(primary_device.device_id, precision)?;
        Ok(Self {
            registration,
            engine,
            partition_inputs,
        })
    }
}

#[cfg(feature = "local-executor")]
impl Worker for LocalEncodeWorkerExecutor {
    fn registration(&self) -> &WorkerRegistration {
        &self.registration
    }
}

#[cfg(feature = "local-executor")]
impl WorkerExecutor for LocalEncodeWorkerExecutor {
    fn execute_task(&self, task: &PartitionTask) -> Result<PartitionTaskResult> {
        runtime_profile_scope!("Worker::ExecuteTask::LocalEncode");

        if task.worker_id != self.registration.worker_id {
            return Err(MahoutError::InvalidInput(format!(
                "task for worker {} was sent to worker {}",
                task.worker_id, self.registration.worker_id
            )));
        }

        let input = self
            .partition_inputs
            .get(&(task.job_id.clone(), task.partition_id))
            .ok_or_else(|| {
                MahoutError::InvalidInput(format!(
                    "missing partition input for job {} partition {}",
                    task.job_id, task.partition_id
                ))
            })?;

        let dlpack_ptr = self
            .engine
            .encode(&input.data, input.num_qubits, &input.encoding_method)?;
        let storage_handle = format!(
            "dlpack:{:p}:state:{}:partition:{}:worker:{}",
            dlpack_ptr, task.state_id, task.partition_id, self.registration.worker_id
        );

        // v1 local executor uses the real qdp-core encode path to validate the control-plane
        // integration, but does not yet persist GPU-resident objects beyond a lightweight handle.
        // Free the DLPack tensor immediately to avoid leaks until the object store lands.
        unsafe {
            free_dlpack_tensor(dlpack_ptr)?;
        }

        Ok(PartitionTaskResult {
            job_id: task.job_id.clone(),
            partition_id: task.partition_id,
            worker_id: self.registration.worker_id.clone(),
            success: true,
            payload: Some(TaskResultPayload::PartitionReady { storage_handle }),
            error: None,
        })
    }
}

#[derive(Default)]
pub struct Coordinator {
    workers: BTreeMap<String, WorkerRegistration>,
    jobs: BTreeMap<String, PlannedJob>,
    task_records: BTreeMap<(String, usize), TaskRecord>,
    objects: BTreeMap<String, RuntimeObjectRecord>,
    retry_policy: RetryPolicy,
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

    pub fn all_devices(&self) -> Vec<DeviceCapabilities> {
        self.cluster_inventory().devices
    }

    pub fn cluster_inventory(&self) -> ClusterInventory {
        ClusterInventory::from_workers(self.workers.values().cloned())
    }

    pub fn device_topology(&self) -> DeviceTopology {
        DeviceTopology::from_inventory(&self.cluster_inventory())
    }

    pub fn plan_job(&mut self, spec: RuntimeJobSpec) -> Result<&PlannedJob> {
        runtime_profile_scope!("Coordinator::PlanJob");

        if spec.job_id.is_empty() {
            return Err(MahoutError::InvalidInput(
                "job_id must not be empty".to_string(),
            ));
        }
        if self.jobs.contains_key(&spec.job_id) {
            return Err(MahoutError::InvalidInput(format!(
                "job {} already exists",
                spec.job_id
            )));
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

        let planned = PlannedJob {
            spec: spec.clone(),
            status: JobStatus::Planned,
            state,
            tasks,
        };

        self.jobs.insert(spec.job_id.clone(), planned);
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
                    let object_id = format!("obj:{}:partition:{}", result.job_id, result.partition_id);
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
                    MetricReduceOp::Min => values
                        .iter()
                        .copied()
                        .reduce(f64::min)
                        .expect("values is non-empty"),
                    MetricReduceOp::Max => values
                        .iter()
                        .copied()
                        .reduce(f64::max)
                        .expect("values is non-empty"),
                    MetricReduceOp::Concat => unreachable!("handled above"),
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

        let worker_ids = workers
            .iter()
            .map(|worker| worker.registration().worker_id.clone())
            .collect::<Vec<_>>();
        if worker_ids.is_empty() {
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
        state: &DistributedStateHandle,
    ) -> Result<Vec<PartitionTask>> {
        runtime_profile_scope!("Coordinator::BuildPartitionTasks");

        let mut worker_by_device = BTreeMap::<(String, usize), String>::new();
        let inventory = self.cluster_inventory();
        for worker in &inventory.workers {
            for device in &worker.devices {
                worker_by_device
                    .insert((device.node_id.clone(), device.device_id), worker.worker_id.clone());
            }
        }

        state.partitions
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
                    record.last_error = Some(format!(
                        "task lease expired after {} attempts",
                        record.attempt_id
                    ));
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

        let job = self.jobs.get_mut(job_id).ok_or_else(|| {
            MahoutError::InvalidInput(format!("unknown job {}", job_id))
        })?;

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

pub struct DistributedRuntimePlanner;

impl DistributedRuntimePlanner {
    pub fn plan_partitions(
        partition_count: usize,
        devices: &[DeviceCapabilities],
        policy: PlacementPolicy,
    ) -> Result<Vec<PartitionAssignment>> {
        runtime_profile_scope!("Runtime::PlanPartitions");

        if devices.is_empty() {
            return Err(MahoutError::InvalidInput(
                "at least one device is required for partition planning".to_string(),
            ));
        }

        let assignments = match policy {
            PlacementPolicy::RoundRobin => Self::plan_round_robin(partition_count, devices),
            PlacementPolicy::Weighted => Self::plan_weighted(partition_count, devices)?,
            PlacementPolicy::TopologyAware => Self::plan_topology_aware(partition_count, devices)?,
        };
        Ok(assignments)
    }

    pub fn build_state_handle(
        state_id: impl Into<String>,
        global_qubits: u32,
        dtype: DType,
        consumption_mode: ConsumptionMode,
        devices: &[DeviceCapabilities],
        policy: PlacementPolicy,
    ) -> Result<DistributedStateHandle> {
        runtime_profile_scope!("Runtime::BuildDistributedState");

        let state_id = state_id.into();
        let partition_count = devices.len().next_power_of_two();
        let layout = PartitionLayout::new(global_qubits, partition_count)?;
        let assignments = Self::plan_partitions(layout.partition_count, devices, policy)?;

        let partitions = assignments
            .into_iter()
            .map(|assignment| StatePartitionRef {
                storage_handle: format!("state:{}:partition:{}", state_id, assignment.partition_id),
                state_id: state_id.clone(),
                partition_id: assignment.partition_id,
                node_id: assignment.node_id,
                device_id: assignment.device_id,
                offset_amplitudes: assignment.partition_id * layout.amplitudes_per_partition,
                amplitude_len: layout.amplitudes_per_partition,
                global_qubits: layout.global_qubits,
                local_qubits: layout.local_qubits,
            })
            .collect();

        Ok(DistributedStateHandle {
            state_id,
            dtype,
            scheme: layout.scheme.clone(),
            consumption_mode,
            layout,
            partitions,
        })
    }

    fn plan_round_robin(
        partition_count: usize,
        devices: &[DeviceCapabilities],
    ) -> Vec<PartitionAssignment> {
        (0..partition_count)
            .map(|partition_id| {
                let device = &devices[partition_id % devices.len()];
                PartitionAssignment {
                    partition_id,
                    node_id: device.node_id.clone(),
                    device_id: device.device_id,
                }
            })
            .collect()
    }

    fn plan_weighted(
        partition_count: usize,
        devices: &[DeviceCapabilities],
    ) -> Result<Vec<PartitionAssignment>> {
        let mut weighted_devices = devices
            .iter()
            .map(|device| {
                Ok(WeightedDevice {
                    node_id: device.node_id.clone(),
                    device_id: device.device_id,
                    weight: device.placement_weight()?,
                    assigned: 0,
                    nvlink_peer_count: device.nvlink_peer_count(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if weighted_devices.iter().all(|d| d.weight <= 0.0) {
            return Err(MahoutError::InvalidInput(
                "all devices have non-positive placement weight".to_string(),
            ));
        }

        let mut assignments = Vec::with_capacity(partition_count);
        for partition_id in 0..partition_count {
            let next_idx = weighted_devices
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    let left_score = left.assigned as f64 / left.weight;
                    let right_score = right.assigned as f64 / right.weight;
                    left_score
                        .total_cmp(&right_score)
                        .then_with(|| left.node_id.cmp(&right.node_id))
                        .then_with(|| left.device_id.cmp(&right.device_id))
                })
                .map(|(idx, _)| idx)
                .expect("weighted_devices is non-empty");

            let device = &mut weighted_devices[next_idx];
            device.assigned += 1;
            assignments.push(PartitionAssignment {
                partition_id,
                node_id: device.node_id.clone(),
                device_id: device.device_id,
            });
        }

        Ok(assignments)
    }

    fn plan_topology_aware(
        partition_count: usize,
        devices: &[DeviceCapabilities],
    ) -> Result<Vec<PartitionAssignment>> {
        runtime_profile_scope!("Runtime::PlanPartitions::TopologyAware");

        let mut weighted_devices = devices
            .iter()
            .map(|device| {
                Ok(WeightedDevice {
                    node_id: device.node_id.clone(),
                    device_id: device.device_id,
                    weight: device.placement_weight()?,
                    assigned: 0,
                    nvlink_peer_count: device.nvlink_peer_count(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if weighted_devices.iter().all(|d| d.weight <= 0.0) {
            return Err(MahoutError::InvalidInput(
                "all devices have non-positive placement weight".to_string(),
            ));
        }

        weighted_devices.sort_by(|left, right| {
            right.nvlink_peer_count
                .cmp(&left.nvlink_peer_count)
                .then_with(|| right.weight.total_cmp(&left.weight))
                .then_with(|| left.node_id.cmp(&right.node_id))
                .then_with(|| left.device_id.cmp(&right.device_id))
        });

        let mut assignments = Vec::with_capacity(partition_count);
        for partition_id in 0..partition_count {
            let next_idx = weighted_devices
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    let left_score = left.assigned as f64 / left.weight;
                    let right_score = right.assigned as f64 / right.weight;
                    left_score
                        .total_cmp(&right_score)
                        .then_with(|| right.nvlink_peer_count.cmp(&left.nvlink_peer_count))
                        .then_with(|| left.node_id.cmp(&right.node_id))
                        .then_with(|| left.device_id.cmp(&right.device_id))
                })
                .map(|(idx, _)| idx)
                .expect("weighted_devices is non-empty");

            let device = &mut weighted_devices[next_idx];
            device.assigned += 1;
            assignments.push(PartitionAssignment {
                partition_id,
                node_id: device.node_id.clone(),
                device_id: device.device_id,
            });
        }

        Ok(assignments)
    }
}

#[derive(Clone, Debug)]
struct WeightedDevice {
    node_id: String,
    device_id: usize,
    weight: f64,
    assigned: usize,
    nvlink_peer_count: usize,
}

#[cfg(test)]
mod tests {
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
        fn execute_task(&self, task: &PartitionTask) -> Result<PartitionTaskResult> {
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
        fn execute_task(&self, task: &PartitionTask) -> Result<PartitionTaskResult> {
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
        assert!(matches!(err, MahoutError::InvalidInput(_)));
    }

    #[test]
    fn weighted_planner_prefers_stronger_device() {
        let devices = vec![
            make_device(
                "node-a",
                0,
                48,
                40,
                40,
                Some(4000.0),
                HostPlatform::Linux,
                1.0,
            ),
            make_device(
                "node-b",
                0,
                24,
                12,
                12,
                Some(1000.0),
                HostPlatform::Wsl,
                0.8,
            ),
        ];

        let assignments =
            DistributedRuntimePlanner::plan_partitions(8, &devices, PlacementPolicy::Weighted)
                .expect("plan");
        let strong = assignments
            .iter()
            .filter(|a| a.node_id == "node-a")
            .count();
        let weak = assignments
            .iter()
            .filter(|a| a.node_id == "node-b")
            .count();

        assert!(strong > weak, "weighted planner should favor stronger device");
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
        let mut gpu0 = make_device(
            "node-a",
            0,
            48,
            40,
            40,
            Some(3000.0),
            HostPlatform::Linux,
            1.0,
        );
        let mut gpu1 = make_device(
            "node-a",
            1,
            48,
            40,
            40,
            Some(3000.0),
            HostPlatform::Linux,
            1.0,
        );
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
        let weak = make_device(
            "node-b",
            0,
            24,
            12,
            12,
            Some(1200.0),
            HostPlatform::Linux,
            1.0,
        );

        let assignments = DistributedRuntimePlanner::plan_partitions(
            4,
            &[gpu0, gpu1, weak],
            PlacementPolicy::TopologyAware,
        )
        .expect("topology-aware plan");

        let nvlink_assignments = assignments
            .iter()
            .filter(|assignment| assignment.node_id == "node-a")
            .count();
        assert!(
            nvlink_assignments >= 3,
            "topology-aware planner should favor NVLink-connected devices for partition placement"
        );
    }

    #[test]
    fn coordinator_registers_workers_and_plans_job() {
        let worker_a = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-a".to_string(),
            node_id: "node-a".to_string(),
            devices: vec![make_device(
                "node-a",
                0,
                48,
                40,
                40,
                Some(3000.0),
                HostPlatform::Linux,
                1.0,
            )],
        })
        .expect("worker-a");

        let worker_b = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-b".to_string(),
            node_id: "node-b".to_string(),
            devices: vec![make_device(
                "node-b",
                0,
                24,
                20,
                20,
                Some(2000.0),
                HostPlatform::Linux,
                1.0,
            )],
        })
        .expect("worker-b");

        let mut coordinator = Coordinator::new();
        coordinator
            .register_worker(&worker_a)
            .expect("register worker-a");
        coordinator
            .register_worker(&worker_b)
            .expect("register worker-b");

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
        assert_eq!(planned.tasks[0].job_id, "job-1");
        assert_eq!(planned.tasks[0].state_id, "state-1");
    }

    #[test]
    fn coordinator_assigns_and_completes_tasks() {
        let worker = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-a".to_string(),
            node_id: "node-a".to_string(),
            devices: vec![make_device(
                "node-a",
                0,
                48,
                40,
                40,
                Some(3000.0),
                HostPlatform::Linux,
                1.0,
            )],
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
            .expect("task should exist");
        assert_eq!(task.partition_id, 0);

        let record = coordinator
            .task_record("job-2", task.partition_id)
            .expect("task record");
        assert_eq!(record.status, TaskStatus::Assigned);
        assert_eq!(record.attempt_id, 1);

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
            .expect("report result");

        let record = coordinator
            .task_record("job-2", task.partition_id)
            .expect("task record after result");
        assert_eq!(record.status, TaskStatus::Completed);
        assert_eq!(
            coordinator.job("job-2").expect("job").status,
            JobStatus::Completed
        );
    }

    #[test]
    fn coordinator_run_loop_completes_job_with_in_process_workers() {
        let worker_a = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-a".to_string(),
            node_id: "node-a".to_string(),
            devices: vec![make_device(
                "node-a",
                0,
                48,
                40,
                40,
                Some(3000.0),
                HostPlatform::Linux,
                1.0,
            )],
        })
        .expect("worker-a");

        let worker_b = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-b".to_string(),
            node_id: "node-b".to_string(),
            devices: vec![make_device(
                "node-b",
                0,
                48,
                40,
                40,
                Some(3000.0),
                HostPlatform::Linux,
                1.0,
            )],
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
            .expect("run job");
        assert_eq!(completed.status, JobStatus::Completed);
        assert!(completed
            .tasks
            .iter()
            .all(|task| coordinator
                .task_record(&task.job_id, task.partition_id)
                .is_some_and(|record| record.status == TaskStatus::Completed)));
    }

    #[test]
    fn coordinator_builds_gather_plan_for_completed_job() {
        let worker_a = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-a".to_string(),
            node_id: "node-a".to_string(),
            devices: vec![make_device(
                "node-a",
                0,
                48,
                40,
                40,
                Some(3000.0),
                HostPlatform::Linux,
                1.0,
            )],
        })
        .expect("worker-a");

        let worker_b = InProcessWorker::new(WorkerRegistration {
            worker_id: "worker-b".to_string(),
            node_id: "node-b".to_string(),
            devices: vec![make_device(
                "node-b",
                0,
                48,
                40,
                40,
                Some(3000.0),
                HostPlatform::Linux,
                1.0,
            )],
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
            .expect("plan job");

        coordinator
            .run_job_with_workers("job-4", &[worker_a, worker_b])
            .expect("run job");

        let gather_plan = coordinator
            .build_gather_plan("job-4", GatherTarget::HostMemory)
            .expect("gather plan");
        assert_eq!(coordinator.objects_for_job("job-4").len(), 2);
        assert!(coordinator.object("obj:job-4:partition:0").is_some());
        assert_eq!(gather_plan.job_id, "job-4");
        assert_eq!(gather_plan.state_id, "state-4");
        assert_eq!(gather_plan.total_amplitudes, 32);
        assert_eq!(gather_plan.segments.len(), 2);
        assert_eq!(gather_plan.segments[0].destination_offset_amplitudes, 0);
        assert_eq!(gather_plan.segments[1].destination_offset_amplitudes, 16);
        assert!(gather_plan
            .segments
            .iter()
            .all(|segment| segment.source_storage_handle.contains("worker")));
    }

    #[test]
    fn coordinator_builds_reduce_plan_for_completed_job() {
        let worker_a = MetricWorker {
            registration: WorkerRegistration {
                worker_id: "worker-a".to_string(),
                node_id: "node-a".to_string(),
                devices: vec![make_device(
                    "node-a",
                    0,
                    48,
                    40,
                    40,
                    Some(3000.0),
                    HostPlatform::Linux,
                    1.0,
                )],
            },
            metric_name: "loss".to_string(),
            value_repr: "1.25".to_string(),
        };

        let worker_b = MetricWorker {
            registration: WorkerRegistration {
                worker_id: "worker-b".to_string(),
                node_id: "node-b".to_string(),
                devices: vec![make_device(
                    "node-b",
                    0,
                    48,
                    40,
                    40,
                    Some(3000.0),
                    HostPlatform::Linux,
                    1.0,
                )],
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
            .expect("plan job");

        coordinator
            .run_job_with_workers("job-5", &[worker_a, worker_b])
            .expect("run job");

        let reduce_plan = coordinator
            .build_reduce_plan("job-5", "loss", MetricReduceOp::Mean)
            .expect("reduce plan");
        assert_eq!(reduce_plan.job_id, "job-5");
        assert_eq!(reduce_plan.metric_name, "loss");
        assert_eq!(reduce_plan.op, MetricReduceOp::Mean);
        assert_eq!(reduce_plan.contributions.len(), 2);
        assert_eq!(reduce_plan.contributions[0].partition_id, 0);
        assert_eq!(reduce_plan.contributions[1].partition_id, 1);

        let reduced = coordinator
            .execute_reduce_plan(&reduce_plan)
            .expect("execute reduce plan");
        assert_eq!(reduced, ReducedMetricValue::Scalar(1.875));
    }

    #[test]
    fn failed_task_requeues_before_max_attempts() {
        let worker = FlakyWorker {
            registration: WorkerRegistration {
                worker_id: "worker-a".to_string(),
                node_id: "node-a".to_string(),
                devices: vec![make_device(
                    "node-a",
                    0,
                    48,
                    40,
                    40,
                    Some(3000.0),
                    HostPlatform::Linux,
                    1.0,
                )],
            },
        };

        let mut coordinator = Coordinator::with_retry_policy(RetryPolicy {
            max_attempts: 2,
            lease_timeout: Duration::from_secs(30),
        });
        coordinator.register_worker(&worker).expect("register worker");
        coordinator
            .plan_job(RuntimeJobSpec {
                job_id: "job-6".to_string(),
                state_id: "state-6".to_string(),
                global_qubits: 4,
                dtype: DType::Complex64,
                consumption_mode: ConsumptionMode::GatherFullState,
                placement_policy: PlacementPolicy::Weighted,
            })
            .expect("plan job");

        let task = coordinator
            .assign_next_task("worker-a")
            .expect("assign task")
            .expect("task");
        coordinator
            .mark_task_running("job-6", task.partition_id, "worker-a")
            .expect("mark running");
        let result = worker.execute_task(&task).expect("execute");
        coordinator.report_task_result(result).expect("report");

        let record = coordinator
            .task_record("job-6", task.partition_id)
            .expect("task record");
        assert_eq!(record.status, TaskStatus::Pending);
        assert_eq!(record.attempt_id, 1);
        assert_eq!(
            coordinator.job("job-6").expect("job").status,
            JobStatus::Planned
        );
    }

    #[test]
    fn inventory_and_topology_resolve_nvlink_peers() {
        let mut device_a0 = make_device(
            "node-a",
            0,
            48,
            40,
            40,
            Some(3000.0),
            HostPlatform::Linux,
            1.0,
        );
        let mut device_a1 = make_device(
            "node-a",
            1,
            48,
            40,
            40,
            Some(3000.0),
            HostPlatform::Linux,
            1.0,
        );
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
                devices: vec![make_device(
                    "node-b",
                    0,
                    24,
                    12,
                    12,
                    Some(1000.0),
                    HostPlatform::Linux,
                    1.0,
                )],
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
