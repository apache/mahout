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

// Tests for basis encoding CUDA kernels.

#![allow(unused_unsafe)]

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
#[cfg(target_os = "linux")]
use qdp_kernels::{
    CuComplex, CuDoubleComplex,
    launch_basis_encode, launch_basis_encode_batch,
    launch_basis_encode_f32, launch_basis_encode_batch_f32,
};

const EPSILON: f64 = 1e-10;
const EPSILON_F32: f32 = 1e-6;

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_first_index() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let state_len = 4usize;
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_basis_encode(
            0,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            state_len,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    assert!((state_h[0].x - 1.0).abs() < EPSILON, "state[0].re should be 1.0");
    assert!(state_h[0].y.abs() < EPSILON, "state[0].im should be 0");
    for (i, item) in state_h.iter().enumerate().skip(1) {
        assert!(item.x.abs() < EPSILON, "state[{i}].re should be 0");
        assert!(item.y.abs() < EPSILON, "state[{i}].im should be 0");
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_last_index() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let state_len = 8usize;
    let basis_index = state_len - 1;
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_basis_encode(
            basis_index,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            state_len,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0);

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    for (i, item) in state_h.iter().enumerate() {
        let expected = if i == basis_index { 1.0 } else { 0.0 };
        assert!((item.x - expected).abs() < EPSILON, "state[{i}].re mismatch");
        assert!(item.y.abs() < EPSILON, "state[{i}].im should be 0");
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_middle_index() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let state_len = 8usize;
    let basis_index = 3usize;
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_basis_encode(
            basis_index,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            state_len,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0);

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    assert!((state_h[3].x - 1.0).abs() < EPSILON, "state[3].re should be 1.0");
    for (i, item) in state_h.iter().enumerate().filter(|&(j, _)| j != 3) {
        assert!(item.x.abs() < EPSILON, "state[{i}].re should be 0");
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_rejects_out_of_range_index() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let state_len = 4usize;
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_basis_encode(
            state_len,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            state_len,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Out-of-range index should be rejected");
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_rejects_zero_state_len() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(1).unwrap();
    let result = unsafe {
        launch_basis_encode(
            0,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            0,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Zero state_len should be rejected");
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_f32_basic() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let state_len = 4usize;
    let basis_index = 2usize;
    let mut state_d = device.alloc_zeros::<CuComplex>(state_len).unwrap();

    let result = unsafe {
        launch_basis_encode_f32(
            basis_index,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            state_len,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0);

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    assert!((state_h[2].x - 1.0f32).abs() < EPSILON_F32, "state[2].re should be 1.0");
    assert!(state_h[2].y.abs() < EPSILON_F32, "state[2].im should be 0");
    for (i, item) in state_h.iter().enumerate().filter(|&(j, _)| j != 2) {
        assert!(item.x.abs() < EPSILON_F32, "state[{i}].re should be 0");
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_f32_rejects_out_of_range() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let state_len = 4usize;
    let mut state_d = device.alloc_zeros::<CuComplex>(state_len).unwrap();

    let result = unsafe {
        launch_basis_encode_f32(
            state_len,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            state_len,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Out-of-range index (f32) should be rejected");
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_batch_basic() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let num_samples = 3usize;
    let state_len = 8usize;
    let num_qubits = 3u32;
    let basis_indices: Vec<usize> = vec![0, 3, 1];

    let indices_d = device.htod_copy(basis_indices.clone()).unwrap();
    let mut state_d = device
        .alloc_zeros::<CuDoubleComplex>(num_samples * state_len)
        .unwrap();

    let result = unsafe {
        launch_basis_encode_batch(
            *indices_d.device_ptr() as *const usize,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            num_samples,
            state_len,
            num_qubits,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0, "Batch basis encode should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    for (sample_idx, &basis_idx) in basis_indices.iter().enumerate() {
        for elem_idx in 0..state_len {
            let expected = if elem_idx == basis_idx { 1.0 } else { 0.0 };
            let actual = state_h[sample_idx * state_len + elem_idx];
            assert!(
                (actual.x - expected).abs() < EPSILON,
                "sample {sample_idx} element {elem_idx}: expected {expected}, got {}",
                actual.x
            );
            assert!(
                actual.y.abs() < EPSILON,
                "sample {sample_idx} element {elem_idx}: imaginary should be 0"
            );
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_batch_rejects_zero_samples() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let indices_d = device.htod_copy(vec![0usize]).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(4).unwrap();

    let result = unsafe {
        launch_basis_encode_batch(
            *indices_d.device_ptr() as *const usize,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            0,
            4,
            2,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Zero num_samples should be rejected");
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_batch_rejects_zero_state_len() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let indices_d = device.htod_copy(vec![0usize]).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(1).unwrap();

    let result = unsafe {
        launch_basis_encode_batch(
            *indices_d.device_ptr() as *const usize,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            1,
            0,
            0,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Zero state_len should be rejected");
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_batch_f32_basic() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let num_samples = 4usize;
    let state_len = 4usize;
    let num_qubits = 2u32;
    let basis_indices: Vec<usize> = vec![0, 1, 2, 3];

    let indices_d = device.htod_copy(basis_indices.clone()).unwrap();
    let mut state_d = device
        .alloc_zeros::<CuComplex>(num_samples * state_len)
        .unwrap();

    let result = unsafe {
        launch_basis_encode_batch_f32(
            *indices_d.device_ptr() as *const usize,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            num_samples,
            state_len,
            num_qubits,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0, "Batch f32 basis encode should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    for (sample_idx, &basis_idx) in basis_indices.iter().enumerate() {
        for elem_idx in 0..state_len {
            let expected = if elem_idx == basis_idx { 1.0f32 } else { 0.0f32 };
            let actual = state_h[sample_idx * state_len + elem_idx];
            assert!(
                (actual.x - expected).abs() < EPSILON_F32,
                "sample {sample_idx} element {elem_idx}: expected {expected}, got {}",
                actual.x
            );
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_basis_encode_batch_f32_rejects_zero_samples() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let indices_d = device.htod_copy(vec![0usize]).unwrap();
    let mut state_d = device.alloc_zeros::<CuComplex>(4).unwrap();

    let result = unsafe {
        launch_basis_encode_batch_f32(
            *indices_d.device_ptr() as *const usize,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            0,
            4,
            2,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Zero num_samples (f32 batch) should be rejected");
}

#[cfg(not(target_os = "linux"))]
#[test]
fn test_basis_encode_dummy_non_linux() {
    let result = unsafe {
        qdp_kernels::launch_basis_encode(
            0,
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 999, "Non-Linux stub should return 999");
}
