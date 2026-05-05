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

use qdp_core::gpu::{CollectiveCommunicator, LocalCollectiveCommunicator};

#[test]
fn local_collective_reduce_sum_returns_total() {
    let comm = LocalCollectiveCommunicator;
    let values = vec![1.0, 2.0, 3.0];
    assert_eq!(comm.all_reduce_sum_f64(&values).unwrap(), 6.0);
}

#[test]
fn local_collective_reduce_sum_rejects_empty_inputs() {
    let comm = LocalCollectiveCommunicator;
    let err = comm.all_reduce_sum_f64(&[]).unwrap_err();
    assert!(matches!(
        err,
        qdp_core::MahoutError::InvalidInput(msg)
        if msg.contains("at least one partial contribution")
    ));
}
