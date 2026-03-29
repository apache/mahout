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

#[cfg(feature = "local-executor")]
use qdp_core::dlpack::free_dlpack_tensor;
use qdp_core::{MahoutError, Result};
#[cfg(feature = "local-executor")]
use qdp_core::{Precision, QdpEngine};

use crate::model::{PartitionInput, PartitionTask, PartitionTaskResult, TaskResultPayload};
use crate::runtime_profile_scope;
use crate::topology::DeviceCapabilities;

#[derive(Clone, Debug, PartialEq)]
pub struct WorkerRegistration {
    pub worker_id: String,
    pub node_id: String,
    pub devices: Vec<DeviceCapabilities>,
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
