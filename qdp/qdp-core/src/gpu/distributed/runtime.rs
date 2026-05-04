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

use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::error::cuda_error_to_string;
use crate::error::{MahoutError, Result};
use crate::gpu::communicator::Communicator;
#[cfg(target_os = "linux")]
use crate::gpu::memory::{BufferStorage, GpuBufferRaw};
use crate::gpu::memory::{Precision, ensure_device_memory_available, map_allocation_error};
use crate::gpu::topology::DeviceMesh;
use crate::gpu::{
    DistributedAmplitudePlan, DistributedStateLayout, DistributedStateVector, PlacementRequest,
};
#[cfg(target_os = "linux")]
use cudarc::driver::{DevicePtr, DevicePtrMut};
#[cfg(target_os = "linux")]
use qdp_kernels::{
    CuComplex, CuDoubleComplex, launch_amplitude_encode, launch_amplitude_encode_f32,
};
#[cfg(target_os = "linux")]
use std::ffi::c_void;

pub(crate) fn validate_distributed_input(
    host_data: &[f64],
    request: &PlacementRequest,
) -> Result<()> {
    if request.num_qubits == 0 {
        return Err(MahoutError::InvalidInput(
            "Number of qubits must be at least 1 for distributed amplitude planning".to_string(),
        ));
    }

    if host_data.is_empty() {
        return Err(MahoutError::InvalidInput(
            "Input data cannot be empty".to_string(),
        ));
    }

    let state_len = request.global_len()?;
    if host_data.len() > state_len {
        return Err(MahoutError::InvalidInput(format!(
            "Input data length {} exceeds state vector size {}",
            host_data.len(),
            state_len
        )));
    }

    Ok(())
}

pub(crate) fn plan_distributed_encode(
    mesh: &DeviceMesh,
    host_data: &[f64],
    request: PlacementRequest,
) -> Result<DistributedAmplitudePlan> {
    validate_distributed_input(host_data, &request)?;
    DistributedAmplitudePlan::for_request(mesh, request)
}

pub(crate) fn calculate_local_norm_sq(
    host_data: &[f64],
    start_idx: usize,
    end_idx: usize,
) -> Result<f64> {
    if start_idx > end_idx {
        return Err(MahoutError::InvalidInput(format!(
            "Invalid shard range: start {} exceeds end {}",
            start_idx, end_idx
        )));
    }

    let slice_end = end_idx.min(host_data.len());
    if start_idx >= slice_end {
        return Ok(0.0);
    }

    let mut local_sum = 0.0f64;
    for &value in &host_data[start_idx..slice_end] {
        if !value.is_finite() {
            return Err(MahoutError::InvalidInput(
                "Input data contains NaN or Inf".to_string(),
            ));
        }
        local_sum += value * value;
    }
    Ok(local_sum)
}

pub(crate) fn calculate_inv_norm_distributed(
    plan: &DistributedAmplitudePlan,
    host_data: &[f64],
    communicator: &dyn Communicator,
) -> Result<f64> {
    let mut partials = Vec::with_capacity(plan.num_devices);
    for shard_id in 0..plan.num_devices {
        let (start_idx, end_idx) = plan.shard_range(shard_id)?;
        partials.push(calculate_local_norm_sq(host_data, start_idx, end_idx)?);
    }

    let global_norm_sq = communicator.reduce_sum_f64(&partials)?;
    if global_norm_sq <= 0.0 || !global_norm_sq.is_finite() {
        return Err(MahoutError::InvalidInput(
            "Input data has zero or non-finite norm (contains NaN, Inf, or all zeros)".to_string(),
        ));
    }

    Ok(1.0 / global_norm_sq.sqrt())
}

pub(crate) fn prepare_distributed_encode(
    mesh: &DeviceMesh,
    host_data: &[f64],
    precision: Precision,
    request: PlacementRequest,
    communicator: &dyn Communicator,
) -> Result<(DistributedAmplitudePlan, f64, DistributedStateLayout)> {
    let plan = plan_distributed_encode(mesh, host_data, request)?;
    let inv_norm = calculate_inv_norm_distributed(&plan, host_data, communicator)?;
    let layout = DistributedStateLayout::new(mesh, &plan, precision)?;
    Ok((plan, inv_norm, layout))
}

#[cfg(target_os = "linux")]
pub(crate) fn encode_distributed_to_shards(
    mesh: &DeviceMesh,
    host_data: &[f64],
    precision: Precision,
    request: PlacementRequest,
    communicator: &dyn Communicator,
) -> Result<DistributedStateVector> {
    let (plan, inv_norm, layout) =
        prepare_distributed_encode(mesh, host_data, precision, request, communicator)?;
    let num_qubits = plan.request.num_qubits;

    let mut buffers = Vec::with_capacity(plan.num_devices);
    for (shard_id, device) in mesh.devices.iter().enumerate() {
        let (start_idx, end_idx) = plan.shard_range(shard_id)?;
        let local_len = end_idx - start_idx;
        let slice_end = end_idx.min(host_data.len());
        let present_len = slice_end.saturating_sub(start_idx);

        let buffer = match precision {
            Precision::Float32 => {
                let requested_bytes = distributed_shard_bytes::<CuComplex>(local_len)?;
                ensure_device_memory_available(
                    requested_bytes,
                    "distributed amplitude shard allocation (f32)",
                    Some(num_qubits),
                    Some(device.ordinal()),
                )?;
                let mut state_slice = device.alloc_zeros::<CuComplex>(local_len).map_err(|e| {
                    map_allocation_error(
                        requested_bytes,
                        "distributed amplitude shard allocation (f32)",
                        Some(num_qubits),
                        Some(device.ordinal()),
                        e,
                    )
                })?;

                if present_len > 0 {
                    let host_input = host_data[start_idx..slice_end]
                        .iter()
                        .map(|&value| value as f32)
                        .collect::<Vec<_>>();
                    let input_slice = device.htod_sync_copy(&host_input).map_err(|e| {
                        MahoutError::MemoryAllocation(format!(
                            "Failed to upload distributed amplitude shard input (f32): {:?}",
                            e
                        ))
                    })?;

                    let ret = unsafe {
                        launch_amplitude_encode_f32(
                            *input_slice.device_ptr() as *const f32,
                            *state_slice.device_ptr_mut() as *mut c_void,
                            present_len,
                            local_len,
                            inv_norm as f32,
                            std::ptr::null_mut(),
                        )
                    };

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(format!(
                            "Distributed amplitude shard kernel failed on cuda:{} with CUDA error code: {} ({})",
                            device.ordinal(),
                            ret,
                            cuda_error_to_string(ret)
                        )));
                    }

                    device.synchronize().map_err(|e| {
                        MahoutError::Cuda(format!(
                            "Distributed amplitude shard synchronize failed on cuda:{}: {:?}",
                            device.ordinal(),
                            e
                        ))
                    })?;
                }

                Arc::new(BufferStorage::F32(GpuBufferRaw { slice: state_slice }))
            }
            Precision::Float64 => {
                let requested_bytes = distributed_shard_bytes::<CuDoubleComplex>(local_len)?;
                ensure_device_memory_available(
                    requested_bytes,
                    "distributed amplitude shard allocation",
                    Some(num_qubits),
                    Some(device.ordinal()),
                )?;
                let mut state_slice =
                    device
                        .alloc_zeros::<CuDoubleComplex>(local_len)
                        .map_err(|e| {
                            map_allocation_error(
                                requested_bytes,
                                "distributed amplitude shard allocation",
                                Some(num_qubits),
                                Some(device.ordinal()),
                                e,
                            )
                        })?;

                if present_len > 0 {
                    let input_slice = device
                        .htod_sync_copy(&host_data[start_idx..slice_end])
                        .map_err(|e| {
                            MahoutError::MemoryAllocation(format!(
                                "Failed to upload distributed amplitude shard input: {:?}",
                                e
                            ))
                        })?;

                    let ret = unsafe {
                        launch_amplitude_encode(
                            *input_slice.device_ptr() as *const f64,
                            *state_slice.device_ptr_mut() as *mut c_void,
                            present_len,
                            local_len,
                            inv_norm,
                            std::ptr::null_mut(),
                        )
                    };

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(format!(
                            "Distributed amplitude shard kernel failed on cuda:{} with CUDA error code: {} ({})",
                            device.ordinal(),
                            ret,
                            cuda_error_to_string(ret)
                        )));
                    }

                    device.synchronize().map_err(|e| {
                        MahoutError::Cuda(format!(
                            "Distributed amplitude shard synchronize failed on cuda:{}: {:?}",
                            device.ordinal(),
                            e
                        ))
                    })?;
                }

                Arc::new(BufferStorage::F64(GpuBufferRaw { slice: state_slice }))
            }
        };

        buffers.push(buffer);
    }

    DistributedStateVector::new_with_buffers(layout, buffers)
}

#[cfg(target_os = "linux")]
fn distributed_shard_bytes<T>(len: usize) -> Result<usize> {
    len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        MahoutError::MemoryAllocation(format!(
            "Distributed shard allocation size overflow (elements={})",
            len
        ))
    })
}
