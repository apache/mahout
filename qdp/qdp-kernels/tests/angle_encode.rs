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
use qdp_kernels::{CuComplex, launch_angle_encode_batch_f32, launch_angle_encode_f32};

const EPSILON_F32: f32 = 1e-5;

#[cfg(target_os = "linux")]
fn expected_amplitude_f32(angles: &[f32], basis_idx: usize) -> f32 {
    angles.iter().enumerate().fold(1.0f32, |acc, (bit, angle)| {
        let factor = if ((basis_idx >> bit) & 1) == 1 {
            angle.sin()
        } else {
            angle.cos()
        };
        acc * factor
    })
}

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

#[test]
#[cfg(target_os = "linux")]
fn test_angle_encode_batch_f32_matches_expected_product_states() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let num_qubits = 3usize;
    let num_samples = 2usize;
    let state_len = 1usize << num_qubits;
    let angles = vec![
        0.0_f32,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::FRAC_PI_4,
        0.2_f32,
        0.4_f32,
        0.6_f32,
    ];

    let input_d = device.htod_copy(angles.clone()).unwrap();
    let mut state_d = device
        .alloc_zeros::<CuComplex>(num_samples * state_len)
        .unwrap();

    let result = unsafe {
        launch_angle_encode_batch_f32(
            *input_d.device_ptr() as *const f32,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            num_samples,
            state_len,
            num_qubits as u32,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0, "Batch kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    for sample_idx in 0..num_samples {
        let sample_angles = &angles[sample_idx * num_qubits..(sample_idx + 1) * num_qubits];
        for basis_idx in 0..state_len {
            let actual = state_h[sample_idx * state_len + basis_idx];
            let expected = expected_amplitude_f32(sample_angles, basis_idx);
            assert!(
                (actual.x - expected).abs() < EPSILON_F32,
                "sample {sample_idx} basis {basis_idx} expected {expected}, got {}",
                actual.x
            );
            assert!(
                actual.y.abs() < EPSILON_F32,
                "sample {sample_idx} basis {basis_idx} imaginary expected 0, got {}",
                actual.y
            );
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_encode_batch_f32_rejects_zero_samples() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let input_d = device.htod_copy(vec![0.0_f32, 1.0_f32]).unwrap();
    let mut state_d = device.alloc_zeros::<CuComplex>(4).unwrap();

    let result = unsafe {
        launch_angle_encode_batch_f32(
            *input_d.device_ptr() as *const f32,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            0,
            4,
            2,
            std::ptr::null_mut(),
        )
    };

    assert_ne!(result, 0, "Zero-sample batch launch should fail");
}
