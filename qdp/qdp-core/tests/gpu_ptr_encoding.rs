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

// Unit and integration tests for encode_from_gpu_ptr and encode_batch_from_gpu_ptr.

#![cfg(target_os = "linux")]

use std::ffi::c_void;

use cudarc::driver::{CudaDevice, DevicePtr};
use qdp_core::{MahoutError, QdpEngine};

mod common;

// ---- Validation / error-path tests (return before using pointer) ----

#[test]
fn test_encode_from_gpu_ptr_unknown_method() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let result = unsafe { engine.encode_from_gpu_ptr(std::ptr::null(), 4, 2, "unknown_encoding") };

    assert!(result.is_err());
    match result {
        Err(MahoutError::NotImplemented(msg)) => {
            assert!(msg.contains("unknown_encoding") || msg.contains("only supports"));
        }
        _ => panic!("expected NotImplemented, got {:?}", result),
    }
}

#[test]
fn test_encode_from_gpu_ptr_amplitude_empty_input() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let result = unsafe { engine.encode_from_gpu_ptr(std::ptr::null(), 0, 2, "amplitude") };

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("empty") || msg.contains("cannot be empty"));
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_encode_from_gpu_ptr_amplitude_input_exceeds_state() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    // 2 qubits -> state_len = 4; request input_len = 10
    let result = unsafe { engine.encode_from_gpu_ptr(std::ptr::null(), 10, 2, "amplitude") };

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("exceeds") && msg.contains("state"));
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_unknown_method() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let result =
        unsafe { engine.encode_batch_from_gpu_ptr(std::ptr::null(), 2, 4, 2, "unknown_method") };

    assert!(result.is_err());
    match result {
        Err(MahoutError::NotImplemented(msg)) => {
            assert!(msg.contains("unknown_method") || msg.contains("only supports"));
        }
        _ => panic!("expected NotImplemented, got {:?}", result),
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_amplitude_num_samples_zero() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let result =
        unsafe { engine.encode_batch_from_gpu_ptr(std::ptr::null(), 0, 4, 2, "amplitude") };

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("zero") || msg.contains("samples"));
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_encode_from_gpu_ptr_basis_input_len_not_one() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Basis single encoding expects exactly 1 value; input_len != 1 must return error.
    let result = unsafe { engine.encode_from_gpu_ptr(std::ptr::null(), 0, 2, "basis") };
    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("exactly 1") || msg.contains("basis"));
        }
        _ => panic!("expected InvalidInput for input_len != 1, got {:?}", result),
    }

    let result = unsafe { engine.encode_from_gpu_ptr(std::ptr::null(), 3, 2, "basis") };
    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("exactly 1") || msg.contains("basis"));
        }
        _ => panic!("expected InvalidInput for input_len != 1, got {:?}", result),
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_basis_sample_size_not_one() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Basis batch expects sample_size == 1 (one index per sample).
    let result =
        unsafe { engine.encode_batch_from_gpu_ptr(std::ptr::null(), 2, 4, 2, "basis") };
    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("sample_size=1") || msg.contains("one index"));
        }
        _ => panic!("expected InvalidInput for sample_size != 1, got {:?}", result),
    }
}

// ---- Happy-path tests (real GPU memory) ----

#[test]
fn test_encode_from_gpu_ptr_amplitude_success() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 4;
    let state_len = 1 << num_qubits;
    let data = common::create_test_data(state_len);

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device");
            return;
        }
    };

    let data_d = match device.htod_sync_copy(data.as_slice()) {
        Ok(b) => b,
        Err(_) => {
            println!("SKIP: Failed to copy to device");
            return;
        }
    };

    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr(ptr, data.len(), num_qubits, "amplitude")
            .expect("encode_from_gpu_ptr should succeed")
    };

    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        let managed = &mut *dlpack_ptr;
        assert!(managed.deleter.is_some(), "Deleter must be present");
        let deleter = managed
            .deleter
            .take()
            .expect("Deleter function pointer is missing");
        deleter(dlpack_ptr);
    }
}

#[test]
fn test_encode_from_gpu_ptr_with_stream_amplitude_success() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 3;
    let state_len = 1 << num_qubits;
    let data = common::create_test_data(state_len);

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device");
            return;
        }
    };

    let data_d = match device.htod_sync_copy(data.as_slice()) {
        Ok(b) => b,
        Err(_) => {
            println!("SKIP: Failed to copy to device");
            return;
        }
    };

    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_with_stream(
                ptr,
                data.len(),
                num_qubits,
                "amplitude",
                std::ptr::null_mut(),
            )
            .expect("encode_from_gpu_ptr_with_stream should succeed")
    };

    assert!(!dlpack_ptr.is_null());

    unsafe {
        let managed = &mut *dlpack_ptr;
        let deleter = managed.deleter.take().expect("Deleter missing");
        deleter(dlpack_ptr);
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_amplitude_success() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 3;
    let state_len = 1 << num_qubits;
    let num_samples = 4;
    let sample_size = state_len;
    let total = num_samples * sample_size;
    let data = common::create_test_data(total);

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device");
            return;
        }
    };

    let data_d = match device.htod_sync_copy(data.as_slice()) {
        Ok(b) => b,
        Err(_) => {
            println!("SKIP: Failed to copy to device");
            return;
        }
    };

    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr(ptr, num_samples, sample_size, num_qubits, "amplitude")
            .expect("encode_batch_from_gpu_ptr should succeed")
    };

    assert!(!dlpack_ptr.is_null());

    unsafe {
        let managed = &mut *dlpack_ptr;
        let deleter = managed.deleter.take().expect("Deleter missing");
        deleter(dlpack_ptr);
    }
}

#[test]
fn test_encode_from_gpu_ptr_basis_success() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 3;
    let basis_index: usize = 0;

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device");
            return;
        }
    };

    let indices: Vec<usize> = vec![basis_index];
    let indices_d = match device.htod_sync_copy(indices.as_slice()) {
        Ok(b) => b,
        Err(_) => {
            println!("SKIP: Failed to copy to device");
            return;
        }
    };

    let ptr = *indices_d.device_ptr() as *const usize as *const c_void;

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr(ptr, 1, num_qubits, "basis")
            .expect("encode_from_gpu_ptr basis should succeed")
    };

    assert!(!dlpack_ptr.is_null());

    unsafe {
        let managed = &mut *dlpack_ptr;
        assert!(managed.deleter.is_some(), "Deleter must be present");
        let deleter = managed.deleter.take().expect("Deleter function pointer is missing");
        deleter(dlpack_ptr);
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_basis_success() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 3;
    let num_samples = 4;
    let sample_size = 1;
    let state_len = 1 << num_qubits;
    let basis_indices: Vec<usize> = (0..num_samples).map(|i| i % state_len).collect();

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device");
            return;
        }
    };

    let indices_d = match device.htod_sync_copy(basis_indices.as_slice()) {
        Ok(b) => b,
        Err(_) => {
            println!("SKIP: Failed to copy to device");
            return;
        }
    };

    let ptr = *indices_d.device_ptr() as *const usize as *const c_void;

    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr(ptr, num_samples, sample_size, num_qubits, "basis")
            .expect("encode_batch_from_gpu_ptr basis should succeed")
    };

    assert!(!dlpack_ptr.is_null());

    unsafe {
        let managed = &mut *dlpack_ptr;
        let deleter = managed.deleter.take().expect("Deleter missing");
        deleter(dlpack_ptr);
    }
}
