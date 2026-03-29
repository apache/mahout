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

use std::time::{Duration, Instant};

use qdp_core::{MahoutError, Result};

use crate::runtime_profile_scope;

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

        let total_amplitudes = 1usize.checked_shl(global_qubits).ok_or_else(|| {
            MahoutError::InvalidInput(format!("global_qubits={} too large", global_qubits))
        })?;
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
