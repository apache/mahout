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

use crate::error::{MahoutError, Result};

/// Abstracts cross-device coordination. PR1 provides a host-coordinated fallback;
/// a later PR can add NCCL-backed implementations behind the same trait.
pub trait Communicator: Send + Sync {
    fn reduce_sum_f64(&self, values: &[f64]) -> Result<f64>;
}

/// Host-coordinated reduction placeholder used for early distributed amplitude prototypes.
#[derive(Default, Debug, Clone, Copy)]
pub struct HostCommunicator;

impl Communicator for HostCommunicator {
    fn reduce_sum_f64(&self, values: &[f64]) -> Result<f64> {
        if values.is_empty() {
            return Err(MahoutError::InvalidInput(
                "Host communicator requires at least one value for reduction".to_string(),
            ));
        }

        Ok(values.iter().copied().sum())
    }
}
