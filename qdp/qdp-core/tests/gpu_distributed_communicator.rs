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

use std::ffi::c_void;

use qdp_core::gpu::{
    CollectiveCommunicator, DeviceCollectiveBackend, DeviceCollectiveCommunicator,
    LocalCollectiveCommunicator, MpiDeviceCollectiveCommunicator, NcclDeviceCollectiveCommunicator,
};

#[test]
fn local_collective_reports_single_rank_identity() {
    let comm = LocalCollectiveCommunicator;

    assert_eq!(comm.rank(), 0);
    assert_eq!(comm.world_size(), 1);
    assert_eq!(comm.all_reduce_sum_f64(12.5).unwrap(), 12.5);
}

#[test]
fn placeholder_mpi_device_collective_is_explicitly_unavailable() {
    let backend = MpiDeviceCollectiveCommunicator;
    assert_eq!(
        backend.backend_kind(),
        DeviceCollectiveBackend::CudaAwareMpi
    );

    let err = unsafe {
        backend.all_reduce_sum_f32_device(
            std::ptr::null::<c_void>(),
            std::ptr::null_mut::<c_void>(),
            0,
            0,
            std::ptr::null_mut::<c_void>(),
        )
    }
    .unwrap_err();

    assert!(matches!(
        err,
        qdp_core::MahoutError::NotImplemented(msg)
        if msg.contains("CUDA-aware MPI device collectives")
    ));
}

#[test]
fn placeholder_nccl_device_collective_is_explicitly_unavailable() {
    let backend = NcclDeviceCollectiveCommunicator;
    assert_eq!(backend.backend_kind(), DeviceCollectiveBackend::Nccl);

    let err = unsafe {
        backend.all_reduce_sum_f32_device(
            std::ptr::null::<c_void>(),
            std::ptr::null_mut::<c_void>(),
            0,
            0,
            std::ptr::null_mut::<c_void>(),
        )
    }
    .unwrap_err();

    assert!(matches!(
        err,
        qdp_core::MahoutError::NotImplemented(msg)
        if msg.contains("NCCL device collectives")
    ));
}
