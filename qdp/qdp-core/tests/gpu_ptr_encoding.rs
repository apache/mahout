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
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use qdp_core::{MahoutError, Precision, QdpEngine};

mod common;

// ---- Helpers for f32 encode_from_gpu_ptr_f32 tests ----

fn engine_f32() -> Option<QdpEngine> {
    QdpEngine::new_with_precision(0, Precision::Float32).ok()
}

fn device_and_f32_slice(data: &[f32]) -> Option<(Arc<CudaDevice>, CudaSlice<f32>)> {
    let device = CudaDevice::new(0).ok()?;
    let slice = device.htod_sync_copy(data).ok()?;
    Some((device, slice))
}

fn assert_dlpack_shape_2_4_and_delete(dlpack_ptr: *mut qdp_core::dlpack::DLManagedTensor) {
    assert!(!dlpack_ptr.is_null());
    unsafe {
        let tensor = &(*dlpack_ptr).dl_tensor;
        assert_eq!(tensor.ndim, 2);
        let shape = std::slice::from_raw_parts(tensor.shape, 2);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 4);
        if let Some(deleter) = (*dlpack_ptr).deleter {
            deleter(dlpack_ptr);
        }
    }
}

// ---- Validation / error-path tests (return before using pointer) ----

#[test]
fn test_encode_from_gpu_ptr_unknown_method() {
    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Need valid GPU pointer so we reach method dispatch (validation runs first)
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let data = common::create_test_data(4);
    let data_d = match device.htod_sync_copy(data.as_slice()) {
        Ok(b) => b,
        Err(_) => return,
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let result = unsafe { engine.encode_from_gpu_ptr(ptr, 4, 2, "unknown_encoding") };

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

    // Need valid GPU pointer so we reach input_len > state_len check (validation runs first)
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let data = common::create_test_data(10);
    let data_d = match device.htod_sync_copy(data.as_slice()) {
        Ok(b) => b,
        Err(_) => return,
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    // 2 qubits -> state_len = 4; request input_len = 10
    let result = unsafe { engine.encode_from_gpu_ptr(ptr, 10, 2, "amplitude") };

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

    // Need valid GPU pointer so we reach method dispatch (validation runs first)
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let data = common::create_test_data(8);
    let data_d = match device.htod_sync_copy(data.as_slice()) {
        Ok(b) => b,
        Err(_) => return,
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let result = unsafe { engine.encode_batch_from_gpu_ptr(ptr, 2, 4, 2, "unknown_method") };

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

    // Need valid GPU pointer so we reach basis input_len checks (validation runs first)
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let indices: Vec<usize> = vec![0, 1, 2];
    let indices_d = match device.htod_sync_copy(indices.as_slice()) {
        Ok(b) => b,
        Err(_) => return,
    };
    let ptr = *indices_d.device_ptr() as *const usize as *const c_void;

    // Basis single encoding expects exactly 1 value; input_len == 0 returns empty error.
    let result = unsafe { engine.encode_from_gpu_ptr(ptr, 0, 2, "basis") };
    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("exactly 1") || msg.contains("basis") || msg.contains("empty"),
                "expected exactly 1 / basis / empty, got: {}",
                msg
            );
        }
        _ => panic!("expected InvalidInput for input_len != 1, got {:?}", result),
    }

    // input_len == 3 (basis expects 1)
    let result = unsafe { engine.encode_from_gpu_ptr(ptr, 3, 2, "basis") };
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

    // Need valid GPU pointer so we reach basis sample_size check (validation runs first)
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let indices: Vec<usize> = vec![0, 1];
    let indices_d = match device.htod_sync_copy(indices.as_slice()) {
        Ok(b) => b,
        Err(_) => return,
    };
    let ptr = *indices_d.device_ptr() as *const usize as *const c_void;

    // Basis batch expects sample_size == 1 (one index per sample); sample_size=4.
    let result = unsafe { engine.encode_batch_from_gpu_ptr(ptr, 2, 4, 2, "basis") };
    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("sample_size=1") || msg.contains("one index"));
        }
        _ => panic!(
            "expected InvalidInput for sample_size != 1, got {:?}",
            result
        ),
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
    // Basis path uses ptr_f64(); engine must be Float64
    let engine = match QdpEngine::new_with_precision(0, Precision::Float64) {
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
        let deleter = managed
            .deleter
            .take()
            .expect("Deleter function pointer is missing");
        deleter(dlpack_ptr);
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_basis_success() {
    // Basis path uses ptr_f64(); engine must be Float64
    let engine = match QdpEngine::new_with_precision(0, Precision::Float64) {
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

// ---- encode_from_gpu_ptr_f32 (float32 amplitude) ----

#[test]
fn test_encode_from_gpu_ptr_f32_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_f32(ptr, input_d.len(), 2)
            .expect("encode_from_gpu_ptr_f32")
    };
    assert_dlpack_shape_2_4_and_delete(dlpack_ptr);
}

#[test]
fn test_encode_from_gpu_ptr_f32_with_stream_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let dlpack_ptr = unsafe {
        engine.encode_from_gpu_ptr_f32_with_stream(ptr, input_d.len(), 2, std::ptr::null_mut())
    }
    .expect("encode_from_gpu_ptr_f32_with_stream");
    assert_dlpack_shape_2_4_and_delete(dlpack_ptr);
}

#[test]
fn test_encode_from_gpu_ptr_f32_with_stream_non_default_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let stream = device.fork_default_stream().expect("fork_default_stream");
    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_f32_with_stream(
                *input_d.device_ptr() as *const f32,
                input_d.len(),
                2,
                stream.stream as *mut c_void,
            )
            .expect("encode_from_gpu_ptr_f32_with_stream (non-default stream)")
    };
    assert_dlpack_shape_2_4_and_delete(dlpack_ptr);
}

#[test]
fn test_encode_from_gpu_ptr_f32_success_f64_engine() {
    let engine = match QdpEngine::new_with_precision(0, Precision::Float64).ok() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_f32(ptr, input_d.len(), 2)
            .expect("encode_from_gpu_ptr_f32 (Float64 engine)")
    };
    assert_dlpack_shape_2_4_and_delete(dlpack_ptr);
}

#[test]
fn test_encode_from_gpu_ptr_f32_empty_input() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let result = unsafe { engine.encode_from_gpu_ptr_f32(ptr, 0, 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("empty")),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_from_gpu_ptr_f32_null_pointer() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let result = unsafe { engine.encode_from_gpu_ptr_f32(std::ptr::null(), 4, 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("null")),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_from_gpu_ptr_f32_input_exceeds_state_len() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let result = unsafe { engine.encode_from_gpu_ptr_f32(ptr, input_d.len(), 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => {
            assert!(
                msg.contains("exceeds") || msg.contains("state vector"),
                "expected 'exceeds' or 'state vector', got: {}",
                msg
            );
        }
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}
