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

//! Direct-launch checks of kernel arithmetic, independent of the Rust
//! descriptors in qdp-core. Each test launches one kernel through the
//! registry with hand-chosen geometry and checks the numbers.
//!
//! Full encoder behaviour (validation, batching, precision) is covered by
//! qdp-core's tests and the Python parity grid.

#![cfg(target_os = "linux")]

use std::sync::Arc;

use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
use qdp_kernels::{CuComplex, CuDoubleComplex, LaunchConfig, config, kernel_args};

fn device() -> Option<Arc<CudaDevice>> {
    if !qdp_kernels::kernels_embedded() {
        println!("SKIP: built without CUDA toolkit");
        return None;
    }
    CudaDevice::new(0).ok().or_else(|| {
        println!("SKIP: no CUDA device available");
        None
    })
}

fn close_f64(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn amplitude_single_kernel_scales_and_zero_pads_odd_input() {
    let Some(device) = device() else { return };
    // 3 inputs into an 8-amplitude state: padding must be exactly zero.
    let input = vec![3.0_f64, 4.0, 0.0];
    let inv_norm = 1.0 / 5.0;
    let state_len = 8usize;
    let input_len = input.len();
    let input_d = device.htod_copy(input).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let f = qdp_kernels::function(&device, "amplitude", "amplitude_encode_kernel").unwrap();
    unsafe {
        qdp_kernels::launch(
            &device,
            f,
            LaunchConfig::grid_1d(state_len.div_ceil(2)),
            std::ptr::null_mut(),
            &mut kernel_args![
                *input_d.device_ptr() as *const f64,
                *state_d.device_ptr_mut() as *mut CuDoubleComplex,
                input_len,
                state_len,
                inv_norm
            ],
        )
        .unwrap();
    }
    device.synchronize().unwrap();

    let host = device.dtoh_sync_copy(&state_d).unwrap();
    let expected = [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for (i, (got, want)) in host.iter().zip(expected).enumerate() {
        assert!(
            close_f64(got.x, want),
            "state[{i}].x = {}, want {want}",
            got.x
        );
        assert!(close_f64(got.y, 0.0), "state[{i}].y = {}", got.y);
    }
}

#[test]
fn l2_norm_batch_kernel_handles_odd_sample_length() {
    let Some(device) = device() else { return };
    // Two samples of length 3: norms 5 and 1 -> inverse norms 0.2 and 1.0.
    let input = vec![3.0_f64, 4.0, 0.0, 0.0, 1.0, 0.0];
    let num_samples = 2usize;
    let sample_len = 3usize;
    let input_d = device.htod_copy(input).unwrap();
    let mut norms_d = device.alloc_zeros::<f64>(num_samples).unwrap();
    let mut flag_d = device.alloc_zeros::<i32>(1).unwrap();
    let blocks_per_sample = 1usize;

    let norm = qdp_kernels::function(&device, "amplitude", "l2_norm_batch_kernel").unwrap();
    let finalize = qdp_kernels::function(&device, "amplitude", "finalize_inv_norm_kernel").unwrap();
    unsafe {
        qdp_kernels::launch(
            &device,
            norm,
            LaunchConfig::exact(
                (num_samples * blocks_per_sample) as u32,
                config::DEFAULT_BLOCK_SIZE as u32,
            ),
            std::ptr::null_mut(),
            &mut kernel_args![
                *input_d.device_ptr() as *const f64,
                num_samples,
                sample_len,
                blocks_per_sample,
                *norms_d.device_ptr_mut() as *mut f64
            ],
        )
        .unwrap();
        qdp_kernels::launch(
            &device,
            finalize,
            LaunchConfig::grid_1d(num_samples),
            std::ptr::null_mut(),
            &mut kernel_args![
                *norms_d.device_ptr_mut() as *mut f64,
                num_samples,
                *flag_d.device_ptr_mut() as *mut i32
            ],
        )
        .unwrap();
    }
    device.synchronize().unwrap();

    let host = device.dtoh_sync_copy(&norms_d).unwrap();
    assert!(close_f64(host[0], 0.2), "inv_norm[0] = {}", host[0]);
    assert!(close_f64(host[1], 1.0), "inv_norm[1] = {}", host[1]);
    assert_eq!(
        device.dtoh_sync_copy(&flag_d).unwrap()[0],
        0,
        "valid norms must not raise the flag"
    );
}

#[test]
fn finalize_inv_norm_kernel_flags_zero_norm_and_writes_zero() {
    let Some(device) = device() else { return };
    // Sums of squares: one valid, one zero.
    let mut norms_d = device.htod_copy(vec![25.0_f64, 0.0]).unwrap();
    let mut flag_d = device.alloc_zeros::<i32>(1).unwrap();
    let count = 2usize;
    let finalize = qdp_kernels::function(&device, "amplitude", "finalize_inv_norm_kernel").unwrap();
    unsafe {
        qdp_kernels::launch(
            &device,
            finalize,
            LaunchConfig::grid_1d(count),
            std::ptr::null_mut(),
            &mut kernel_args![
                *norms_d.device_ptr_mut() as *mut f64,
                count,
                *flag_d.device_ptr_mut() as *mut i32
            ],
        )
        .unwrap();
    }
    device.synchronize().unwrap();
    let host = device.dtoh_sync_copy(&norms_d).unwrap();
    assert!(close_f64(host[0], 0.2) && host[1] == 0.0, "{host:?}");
    assert_eq!(device.dtoh_sync_copy(&flag_d).unwrap()[0], 1);
}

#[test]
fn amplitude_batch_kernel_handles_misaligned_odd_samples() {
    let Some(device) = device() else { return };
    // sample_size 3 puts every second sample at an address that is not
    // 16-byte aligned, exercising the scalar fallback next to the double2 load.
    let input = vec![3.0_f64, 4.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let inv_norms = vec![0.2_f64, 1.0, 1.0 / 3.0_f64.sqrt()];
    let num_samples = 3usize;
    let sample_size = 3usize;
    let state_len = 4usize;
    let input_d = device.htod_copy(input).unwrap();
    let norms_d = device.htod_copy(inv_norms.clone()).unwrap();
    let mut state_d = device
        .alloc_zeros::<CuDoubleComplex>(num_samples * state_len)
        .unwrap();
    let f = qdp_kernels::function(&device, "amplitude", "amplitude_encode_batch_kernel").unwrap();
    unsafe {
        qdp_kernels::launch(
            &device,
            f,
            LaunchConfig::grid_stride(num_samples * (state_len / 2), config::MAX_GRID_BLOCKS),
            std::ptr::null_mut(),
            &mut kernel_args![
                *input_d.device_ptr() as *const f64,
                *state_d.device_ptr_mut() as *mut CuDoubleComplex,
                *norms_d.device_ptr() as *const f64,
                num_samples,
                sample_size,
                state_len
            ],
        )
        .unwrap();
    }
    device.synchronize().unwrap();
    let host = device.dtoh_sync_copy(&state_d).unwrap();
    let s3 = 1.0 / 3.0_f64.sqrt();
    let expected = [0.6, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, s3, s3, s3, 0.0];
    for (i, (got, want)) in host.iter().zip(expected).enumerate() {
        assert!(
            close_f64(got.x, want),
            "state[{i}].x = {}, want {want}",
            got.x
        );
        assert!(close_f64(got.y, 0.0), "state[{i}].y = {}", got.y);
    }
}

#[test]
fn angle_batch_kernel_f32_builds_product_states() {
    let Some(device) = device() else { return };
    // Sample 0: angles (π/2, 0) -> |1>|0> = index 1. Sample 1: (0, π/2) -> index 2.
    let angles = vec![
        std::f32::consts::FRAC_PI_2,
        0.0_f32,
        0.0,
        std::f32::consts::FRAC_PI_2,
    ];
    let num_samples = 2usize;
    let num_qubits = 2u32;
    let state_len = 4usize;
    let angles_d = device.htod_copy(angles).unwrap();
    let mut state_d = device
        .alloc_zeros::<CuComplex>(num_samples * state_len)
        .unwrap();

    let f = qdp_kernels::function(&device, "angle", "angle_encode_batch_kernel_f32").unwrap();
    unsafe {
        qdp_kernels::launch(
            &device,
            f,
            LaunchConfig::grid_stride(num_samples * state_len, config::MAX_GRID_BLOCKS),
            std::ptr::null_mut(),
            &mut kernel_args![
                *angles_d.device_ptr() as *const f32,
                *state_d.device_ptr_mut() as *mut CuComplex,
                num_samples,
                state_len,
                num_qubits
            ],
        )
        .unwrap();
    }
    device.synchronize().unwrap();

    let host = device.dtoh_sync_copy(&state_d).unwrap();
    let expected = [0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    for (i, (got, want)) in host.iter().zip(expected).enumerate() {
        assert!(
            (got.x - want).abs() < 1e-6,
            "state[{i}].x = {}, want {want}",
            got.x
        );
        assert!(got.y.abs() < 1e-6, "state[{i}].y = {}", got.y);
    }
}

#[test]
fn check_finite_kernel_f32_flags_nan_and_inf_only() {
    let Some(device) = device() else { return };
    let cases: [(&str, Vec<f32>, i32); 3] = [
        ("clean", vec![1.0, -2.0, 3.5, 0.0], 0),
        ("nan", vec![1.0, f32::NAN, 3.5, 0.0], 1),
        ("inf", vec![1.0, 2.0, f32::INFINITY, 0.0], 1),
    ];
    let f = qdp_kernels::function(&device, "validation", "check_finite_batch_kernel_f32").unwrap();
    for (label, values, want) in cases {
        let total = values.len();
        let input_d = device.htod_copy(values).unwrap();
        let mut flag_d = device.alloc_zeros::<i32>(1).unwrap();
        unsafe {
            qdp_kernels::launch(
                &device,
                f,
                LaunchConfig::grid_stride(total, config::MAX_GRID_BLOCKS),
                std::ptr::null_mut(),
                &mut kernel_args![
                    *input_d.device_ptr() as *const f32,
                    total,
                    *flag_d.device_ptr_mut() as *mut i32
                ],
            )
            .unwrap();
        }
        device.synchronize().unwrap();
        let flag = device.dtoh_sync_copy(&flag_d).unwrap()[0];
        assert_eq!(flag != 0, want != 0, "{label}: flag = {flag}");
    }
}
