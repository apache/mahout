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

// Tests for angle encoding CUDA kernels.

#![allow(unused_unsafe)]

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
#[cfg(target_os = "linux")]
use qdp_kernels::{CuComplex, launch_angle_encode_f32};

const EPSILON_F32: f32 = 1e-5;

#[test]
#[cfg(target_os = "linux")]
fn test_angle_encode_basic_f32() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let input = vec![std::f32::consts::FRAC_PI_2, 0.0_f32];
    let state_len = 4usize;

    let input_d = device.htod_copy(input).unwrap();
    let mut state_d = device.alloc_zeros::<CuComplex>(state_len).unwrap();

    let result = unsafe {
        launch_angle_encode_f32(
            *input_d.device_ptr() as *const f32,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            state_len,
            2,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    let expected = [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32];

    for (idx, (actual, expected)) in state_h.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual.x - expected).abs() < EPSILON_F32,
            "state[{idx}].x expected {expected}, got {}",
            actual.x
        );
        assert!(
            actual.y.abs() < EPSILON_F32,
            "state[{idx}].y expected 0, got {}",
            actual.y
        );
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_encode_matches_expected_product_state_f32() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let angles = vec![0.3_f32, 0.7_f32];
    let state_len = 4usize;

    let input_d = device.htod_copy(angles.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuComplex>(state_len).unwrap();

    let result = unsafe {
        launch_angle_encode_f32(
            *input_d.device_ptr() as *const f32,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            state_len,
            2,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    let expected = [
        angles[0].cos() * angles[1].cos(),
        angles[0].sin() * angles[1].cos(),
        angles[0].cos() * angles[1].sin(),
        angles[0].sin() * angles[1].sin(),
    ];

    for (idx, (actual, expected)) in state_h.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual.x - expected).abs() < EPSILON_F32,
            "state[{idx}].x expected {expected}, got {}",
            actual.x
        );
        assert!(
            actual.y.abs() < EPSILON_F32,
            "state[{idx}].y expected 0, got {}",
            actual.y
        );
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_encode_f32_rejects_zero_qubits() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let input = vec![0.0_f32];
    let input_d = device.htod_copy(input).unwrap();
    let mut state_d = device.alloc_zeros::<CuComplex>(1).unwrap();

    let result = unsafe {
        launch_angle_encode_f32(
            *input_d.device_ptr() as *const f32,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            1,
            0,
            std::ptr::null_mut(),
        )
    };

    assert_ne!(result, 0, "Zero-qubit launch should fail");
}
