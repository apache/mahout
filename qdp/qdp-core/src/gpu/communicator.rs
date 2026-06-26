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

use crate::error::{MahoutError, Result};

/// Abstracts cross-shard collective operations.
///
/// Implementations expose rank-local scalar semantics: each rank contributes
/// one local value and receives the globally reduced scalar.
pub trait CollectiveCommunicator: Send + Sync {
    /// Rank of the current process in the collective world.
    fn rank(&self) -> usize;

    /// Number of ranks participating in the collective world.
    fn world_size(&self) -> usize;

    /// Sum one rank-local contribution into one global scalar.
    fn all_reduce_sum_f64(&self, local_value: f64) -> Result<f64>;
}

/// Device collective backend selected for GPU-resident reductions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceCollectiveBackend {
    Local,
    CudaAwareMpi,
    Nccl,
    Unavailable,
}

/// Abstracts GPU-resident collective operations.
pub trait DeviceCollectiveCommunicator: Send + Sync {
    fn backend_kind(&self) -> DeviceCollectiveBackend;

    /// # Safety
    ///
    /// Callers must ensure the raw pointers and stream are valid for the
    /// selected CUDA device and that `recv_ptr` can hold `count` `f32` values.
    unsafe fn all_reduce_sum_f32_device(
        &self,
        send_ptr: *const c_void,
        recv_ptr: *mut c_void,
        count: usize,
        device_id: usize,
        stream: *mut c_void,
    ) -> Result<()>;
}

/// In-process collective implementation for the single-rank path.
#[derive(Default, Debug, Clone, Copy)]
pub struct LocalCollectiveCommunicator;

impl CollectiveCommunicator for LocalCollectiveCommunicator {
    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }

    fn all_reduce_sum_f64(&self, local_value: f64) -> Result<f64> {
        Ok(local_value)
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct MpiDeviceCollectiveCommunicator;

impl DeviceCollectiveCommunicator for MpiDeviceCollectiveCommunicator {
    fn backend_kind(&self) -> DeviceCollectiveBackend {
        DeviceCollectiveBackend::CudaAwareMpi
    }

    unsafe fn all_reduce_sum_f32_device(
        &self,
        _send_ptr: *const c_void,
        _recv_ptr: *mut c_void,
        _count: usize,
        _device_id: usize,
        _stream: *mut c_void,
    ) -> Result<()> {
        Err(MahoutError::NotImplemented(
            "CUDA-aware MPI device collectives are reserved but not implemented".to_string(),
        ))
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct NcclDeviceCollectiveCommunicator;

impl DeviceCollectiveCommunicator for NcclDeviceCollectiveCommunicator {
    fn backend_kind(&self) -> DeviceCollectiveBackend {
        DeviceCollectiveBackend::Nccl
    }

    unsafe fn all_reduce_sum_f32_device(
        &self,
        _send_ptr: *const c_void,
        _recv_ptr: *mut c_void,
        _count: usize,
        _device_id: usize,
        _stream: *mut c_void,
    ) -> Result<()> {
        Err(MahoutError::NotImplemented(
            "NCCL device collectives are reserved but not implemented".to_string(),
        ))
    }
}
