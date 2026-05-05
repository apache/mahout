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

/// Abstracts cross-shard collective operations.
///
/// The current implementation executes collectives inside one process. A future
/// MPI-backed implementation can provide the same interface while mapping the
/// partial contributions to rank-local shards and performing a real all-reduce.
pub trait CollectiveCommunicator: Send + Sync {
    /// Sum one set of per-shard partial contributions into one global scalar.
    fn all_reduce_sum_f64(&self, values: &[f64]) -> Result<f64>;
}

/// In-process collective implementation for the current single-process
/// distributed path.
#[derive(Default, Debug, Clone, Copy)]
pub struct LocalCollectiveCommunicator;

impl CollectiveCommunicator for LocalCollectiveCommunicator {
    fn all_reduce_sum_f64(&self, values: &[f64]) -> Result<f64> {
        if values.is_empty() {
            return Err(MahoutError::InvalidInput(
                "Collective reduction requires at least one partial contribution".to_string(),
            ));
        }

        Ok(values.iter().copied().sum())
    }
}
