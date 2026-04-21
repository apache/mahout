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

use cudarc::driver::{DevicePtr, DeviceSlice};
use qdp_core::{MahoutError, Precision, QdpEngine};
use std::ffi::c_void;

mod common;

/// IQP full encoding expected data length: n + n*(n-1)/2.
fn iqp_full_data_len(num_qubits: usize) -> usize {
    num_qubits + num_qubits * (num_qubits.saturating_sub(1)) / 2
}

/// IQP-Z encoding expected data length: n.
fn iqp_z_data_len(num_qubits: usize) -> usize {
    num_qubits
}

// ---- Helpers for f32 encode_from_gpu_ptr_f32 tests ----

fn engine_f32() -> Option<QdpEngine> {
    common::qdp_engine_with_precision(Precision::Float32)
}

// ---- Validation / error-path tests (return before using pointer) ----

#[test]
fn test_encode_from_gpu_ptr_unknown_method() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // Need valid GPU pointer so we reach method dispatch (validation runs first)
    let data = common::create_test_data(4);
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let result = unsafe { engine.encode_from_gpu_ptr(ptr, 4, 2, "unknown_encoding") };

    assert!(result.is_err());
    match result {
        Err(MahoutError::NotImplemented(msg)) => {
            assert!(msg.contains("unknown_encoding") || msg.contains("only supports"));
        }
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("Unknown encoder") || msg.contains("unknown_encoding"));
        }
        _ => panic!("expected NotImplemented or InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_encode_from_gpu_ptr_amplitude_empty_input() {
    let Some(engine) = common::qdp_engine() else {
        return;
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
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // Need valid GPU pointer so we reach input_len > state_len check (validation runs first)
    let data = common::create_test_data(10);
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
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
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // Need valid GPU pointer so we reach method dispatch (validation runs first)
    let data = common::create_test_data(8);
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let result = unsafe { engine.encode_batch_from_gpu_ptr(ptr, 2, 4, 2, "unknown_method") };

    assert!(result.is_err());
    match result {
        Err(MahoutError::NotImplemented(msg)) => {
            assert!(msg.contains("unknown_method") || msg.contains("only supports"));
        }
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("Unknown encoder") || msg.contains("unknown_method"));
        }
        _ => panic!("expected NotImplemented or InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_amplitude_num_samples_zero() {
    let Some(engine) = common::qdp_engine() else {
        return;
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
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // Need valid GPU pointer so we reach basis input_len checks (validation runs first)
    let indices: Vec<usize> = vec![0, 1, 2];
    let Some((_device, indices_d)) = common::copy_usize_to_device(indices.as_slice()) else {
        return;
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
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // Need valid GPU pointer so we reach basis sample_size check (validation runs first)
    let indices: Vec<usize> = vec![0, 1];
    let Some((_device, indices_d)) = common::copy_usize_to_device(indices.as_slice()) else {
        return;
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
    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 4;
    let state_len = 1 << num_qubits;
    let data = common::create_test_data(state_len);

    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        println!("SKIP: Failed to copy to device");
        return;
    };

    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr(ptr, data.len(), num_qubits, "amplitude")
            .expect("encode_from_gpu_ptr should succeed")
    };

    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        common::take_deleter_and_delete(dlpack_ptr);
    }
}

#[test]
fn test_encode_from_gpu_ptr_with_stream_amplitude_success() {
    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 3;
    let state_len = 1 << num_qubits;
    let data = common::create_test_data(state_len);

    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        println!("SKIP: Failed to copy to device");
        return;
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
        common::take_deleter_and_delete(dlpack_ptr);
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_amplitude_success() {
    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 3;
    let state_len = 1 << num_qubits;
    let num_samples = 4;
    let sample_size = state_len;
    let total = num_samples * sample_size;
    let data = common::create_test_data(total);

    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        println!("SKIP: Failed to copy to device");
        return;
    };

    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr(ptr, num_samples, sample_size, num_qubits, "amplitude")
            .expect("encode_batch_from_gpu_ptr should succeed")
    };

    assert!(!dlpack_ptr.is_null());

    unsafe {
        common::take_deleter_and_delete(dlpack_ptr);
    }
}

#[test]
fn test_encode_from_gpu_ptr_basis_success() {
    // Basis path uses ptr_f64(); engine must be Float64
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 3;
    let basis_index: usize = 0;

    let indices: Vec<usize> = vec![basis_index];
    let Some((_device, indices_d)) = common::copy_usize_to_device(indices.as_slice()) else {
        println!("SKIP: Failed to copy to device");
        return;
    };

    let ptr = *indices_d.device_ptr() as *const usize as *const c_void;

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr(ptr, 1, num_qubits, "basis")
            .expect("encode_from_gpu_ptr basis should succeed")
    };

    assert!(!dlpack_ptr.is_null());

    unsafe {
        common::take_deleter_and_delete(dlpack_ptr);
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_basis_success() {
    // Basis path uses ptr_f64(); engine must be Float64
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 3;
    let num_samples = 4;
    let sample_size = 1;
    let state_len = 1 << num_qubits;
    let basis_indices: Vec<usize> = (0..num_samples).map(|i| i % state_len).collect();

    let Some((_device, indices_d)) = common::copy_usize_to_device(basis_indices.as_slice()) else {
        println!("SKIP: Failed to copy to device");
        return;
    };

    let ptr = *indices_d.device_ptr() as *const usize as *const c_void;

    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr(ptr, num_samples, sample_size, num_qubits, "basis")
            .expect("encode_batch_from_gpu_ptr basis should succeed")
    };

    assert!(!dlpack_ptr.is_null());

    unsafe {
        common::take_deleter_and_delete(dlpack_ptr);
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_iqp_success() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 2;
    let state_len = 1 << num_qubits;
    let sample_size = iqp_full_data_len(num_qubits);
    let num_samples = 3;
    let total = num_samples * sample_size;
    let data: Vec<f64> = (0..total).map(|i| (i as f64) * 0.05).collect();
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr(ptr, num_samples, sample_size, num_qubits, "iqp")
            .expect("encode_batch_from_gpu_ptr iqp should succeed")
    };
    assert!(!dlpack_ptr.is_null());
    unsafe {
        common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, num_samples as i64, state_len as i64)
    };
}

#[test]
fn test_encode_batch_from_gpu_ptr_iqp_z_success() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 2;
    let state_len = 1 << num_qubits;
    let sample_size = iqp_z_data_len(num_qubits);
    let num_samples = 3;
    let total = num_samples * sample_size;
    let data: Vec<f64> = (0..total).map(|i| (i as f64) * 0.05).collect();
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr(ptr, num_samples, sample_size, num_qubits, "iqp-z")
            .expect("encode_batch_from_gpu_ptr iqp-z should succeed")
    };
    assert!(!dlpack_ptr.is_null());
    unsafe {
        common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, num_samples as i64, state_len as i64)
    };
}

#[test]
fn test_encode_batch_from_gpu_ptr_iqp_wrong_sample_size() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 2;
    let expected_sample_size = iqp_full_data_len(num_qubits);
    let wrong_sample_size = expected_sample_size + 1;
    let num_samples = 2;
    let data = vec![0.1_f64; num_samples * wrong_sample_size];
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let result = unsafe {
        engine.encode_batch_from_gpu_ptr(ptr, num_samples, wrong_sample_size, num_qubits, "iqp")
    };
    assert!(result.is_err());
    match &result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("expects") || msg.contains("sample_size"),
                "msg: {}",
                msg
            );
        }
        _ => panic!("expected InvalidInput"),
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_iqp_z_wrong_sample_size() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 2;
    let expected_sample_size = iqp_z_data_len(num_qubits);
    let wrong_sample_size = expected_sample_size + 1;
    let num_samples = 2;
    let data = vec![0.1_f64; num_samples * wrong_sample_size];
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let result = unsafe {
        engine.encode_batch_from_gpu_ptr(ptr, num_samples, wrong_sample_size, num_qubits, "iqp-z")
    };
    assert!(result.is_err());
    match &result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("expects") || msg.contains("sample_size"),
                "msg: {}",
                msg
            );
        }
        _ => panic!("expected InvalidInput"),
    }
}

#[test]
fn test_encode_from_gpu_ptr_iqp_z_success() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 2;
    let data = [0.1_f64, -0.2_f64];

    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        println!("SKIP: Failed to copy to device");
        return;
    };

    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr(ptr, data.len(), num_qubits, "iqp-z")
            .expect("encode_from_gpu_ptr iqp-z should succeed")
    };

    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
}

#[test]
fn test_encode_from_gpu_ptr_iqp_success() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 2;
    let data = [0.1_f64, -0.2_f64, 0.3_f64];

    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        println!("SKIP: Failed to copy to device");
        return;
    };

    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr(ptr, data.len(), num_qubits, "iqp")
            .expect("encode_from_gpu_ptr iqp should succeed")
    };

    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
}

#[test]
fn test_encode_from_gpu_ptr_iqp_wrong_input_len() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 2;
    let expected_len = iqp_full_data_len(num_qubits);
    let data = vec![0.1_f64; expected_len];
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let result_too_few =
        unsafe { engine.encode_from_gpu_ptr(ptr, expected_len - 1, num_qubits, "iqp") };
    assert!(result_too_few.is_err());
    match &result_too_few {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("expects") || msg.contains("sample"))
        }
        _ => panic!("expected InvalidInput"),
    }

    let result_too_many =
        unsafe { engine.encode_from_gpu_ptr(ptr, expected_len + 1, num_qubits, "iqp") };
    assert!(result_too_many.is_err());
}

#[test]
fn test_encode_from_gpu_ptr_iqp_z_wrong_input_len() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 2;
    let expected_len = iqp_z_data_len(num_qubits);
    let data = vec![0.1_f64; expected_len];
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;

    let result = unsafe { engine.encode_from_gpu_ptr(ptr, expected_len + 1, num_qubits, "iqp-z") };
    assert!(result.is_err());
    match &result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("expects") || msg.contains("sample"))
        }
        _ => panic!("expected InvalidInput"),
    }
}

#[test]
fn test_encode_from_gpu_ptr_with_stream_iqp_success() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 2;
    let data = [0.1_f64, -0.2_f64, 0.3_f64];
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_with_stream(
                ptr,
                data.len(),
                num_qubits,
                "iqp",
                std::ptr::null_mut(),
            )
            .expect("encode_from_gpu_ptr_with_stream iqp")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
}

#[test]
fn test_encode_from_gpu_ptr_with_stream_iqp_z_success() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 2;
    let data = [0.1_f64, -0.2_f64];
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_with_stream(
                ptr,
                data.len(),
                num_qubits,
                "iqp-z",
                std::ptr::null_mut(),
            )
            .expect("encode_from_gpu_ptr_with_stream iqp-z")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
}

#[test]
fn test_encode_from_gpu_ptr_iqp_three_qubits() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 3;
    let state_len = 1 << num_qubits;
    let expected_len = iqp_full_data_len(num_qubits);
    let data: Vec<f64> = (0..expected_len).map(|i| (i as f64) * 0.1).collect();
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr(ptr, data.len(), num_qubits, "iqp")
            .expect("encode_from_gpu_ptr iqp 3 qubits")
    };
    assert!(!dlpack_ptr.is_null());
    unsafe {
        let tensor = &(*dlpack_ptr).dl_tensor;
        assert_eq!(tensor.ndim, 2);
        let shape = std::slice::from_raw_parts(tensor.shape, 2);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], state_len as i64);
        if let Some(deleter) = (*dlpack_ptr).deleter {
            deleter(dlpack_ptr);
        }
    }
}

#[test]
fn test_encode_from_gpu_ptr_iqp_z_three_qubits() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        return;
    };
    let num_qubits = 3;
    let state_len = 1 << num_qubits;
    let expected_len = iqp_z_data_len(num_qubits);
    let data: Vec<f64> = (0..expected_len).map(|i| (i as f64) * 0.1).collect();
    let Some((_device, data_d)) = common::copy_f64_to_device(data.as_slice()) else {
        return;
    };
    let ptr = *data_d.device_ptr() as *const f64 as *const c_void;
    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr(ptr, data.len(), num_qubits, "iqp-z")
            .expect("encode_from_gpu_ptr iqp-z 3 qubits")
    };
    assert!(!dlpack_ptr.is_null());
    unsafe {
        let tensor = &(*dlpack_ptr).dl_tensor;
        assert_eq!(tensor.ndim, 2);
        let shape = std::slice::from_raw_parts(tensor.shape, 2);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], state_len as i64);
        if let Some(deleter) = (*dlpack_ptr).deleter {
            deleter(dlpack_ptr);
        }
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
    let (_device, input_d) = match common::copy_f32_to_device(&[1.0, 0.0, 0.0, 0.0]) {
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
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
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
    let (_device, input_d) = match common::copy_f32_to_device(&[1.0, 0.0, 0.0, 0.0]) {
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
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
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
    let (device, input_d) = match common::copy_f32_to_device(&[1.0, 0.0, 0.0, 0.0]) {
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
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
}

#[test]
fn test_encode_from_gpu_ptr_f32_success_f64_engine() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        println!("SKIP: No GPU");
        return;
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[1.0, 0.0, 0.0, 0.0]) {
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
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
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
    let (_device, input_d) = match common::copy_f32_to_device(&[1.0]) {
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
    let (_device, input_d) = match common::copy_f32_to_device(&[1.0, 0.0, 0.0, 0.0, 0.0]) {
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

#[test]
fn test_encode_angle_from_gpu_ptr_f32_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0, std::f32::consts::FRAC_PI_2]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let dlpack_ptr = unsafe {
        engine
            .encode_angle_from_gpu_ptr_f32(ptr, input_d.len(), 2)
            .expect("encode_angle_from_gpu_ptr_f32")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
}

#[test]
fn test_encode_angle_from_gpu_ptr_f32_with_stream_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (device, input_d) = match common::copy_f32_to_device(&[0.0, std::f32::consts::FRAC_PI_2]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let stream = device.fork_default_stream().expect("fork_default_stream");
    let dlpack_ptr = unsafe {
        engine
            .encode_angle_from_gpu_ptr_f32_with_stream(
                *input_d.device_ptr() as *const f32,
                input_d.len(),
                2,
                stream.stream as *mut c_void,
            )
            .expect("encode_angle_from_gpu_ptr_f32_with_stream")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
}

#[test]
fn test_encode_angle_from_gpu_ptr_f32_success_f64_engine() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        println!("SKIP: No GPU");
        return;
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0, std::f32::consts::FRAC_PI_2]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let dlpack_ptr = unsafe {
        engine
            .encode_angle_from_gpu_ptr_f32(ptr, input_d.len(), 2)
            .expect("encode_angle_from_gpu_ptr_f32 (Float64 engine)")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 4) };
}

#[test]
fn test_encode_angle_from_gpu_ptr_f32_empty_input() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let result = unsafe { engine.encode_angle_from_gpu_ptr_f32(ptr, 0, 1) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => {
            assert!(msg.contains("empty") || msg.contains("null"));
        }
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_angle_from_gpu_ptr_f32_null_pointer() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let result = unsafe { engine.encode_angle_from_gpu_ptr_f32(std::ptr::null(), 2, 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("null")),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_angle_from_gpu_ptr_f32_qubit_mismatch() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0, std::f32::consts::FRAC_PI_2]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let result = unsafe { engine.encode_angle_from_gpu_ptr_f32(ptr, input_d.len(), 1) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => {
            assert!(msg.contains("expects 1 values") || msg.contains("got 2"));
        }
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_angle_from_gpu_ptr_f32_too_many_qubits() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let input = vec![0.0_f32; 31];
    let (_device, input_d) = match common::copy_f32_to_device(&input) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let result = unsafe { engine.encode_angle_from_gpu_ptr_f32(ptr, input_d.len(), 31) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => {
            assert!(msg.contains("exceeds practical limit"), "got: {msg}");
        }
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_angle_from_gpu_ptr_f32_with_stream_too_many_qubits() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (device, input_d) = match common::copy_f32_to_device(&[0.0_f32; 31]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let stream = device.fork_default_stream().expect("fork_default_stream");
    let result = unsafe {
        engine.encode_angle_from_gpu_ptr_f32_with_stream(
            *input_d.device_ptr() as *const f32,
            input_d.len(),
            31,
            stream.stream as *mut c_void,
        )
    };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => {
            assert!(msg.contains("exceeds practical limit"), "got: {msg}");
        }
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_f32_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let num_samples = 2;
    let sample_size = 4;
    let (_device, input_d) =
        match common::copy_f32_to_device(&[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]) {
            Some(t) => t,
            None => {
                println!("SKIP: No CUDA device");
                return;
            }
        };
    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr_f32(
                *input_d.device_ptr() as *const f32,
                num_samples,
                sample_size,
                2,
            )
            .expect("encode_batch_from_gpu_ptr_f32")
    };
    unsafe {
        common::assert_dlpack_shape_2d_and_delete(
            dlpack_ptr,
            num_samples as i64,
            sample_size as i64,
        )
    };
}

#[test]
fn test_encode_batch_from_gpu_ptr_f32_with_stream_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (device, input_d) =
        match common::copy_f32_to_device(&[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]) {
            Some(t) => t,
            None => {
                println!("SKIP: No CUDA device");
                return;
            }
        };
    let stream = device.fork_default_stream().expect("fork_default_stream");
    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr_f32_with_stream(
                *input_d.device_ptr() as *const f32,
                2,
                4,
                2,
                stream.stream as *mut c_void,
            )
            .expect("encode_batch_from_gpu_ptr_f32_with_stream")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 2, 4) };
}

#[test]
fn test_encode_batch_from_gpu_ptr_f32_success_f64_engine() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        println!("SKIP: No GPU");
        return;
    };
    let (_device, input_d) =
        match common::copy_f32_to_device(&[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]) {
            Some(t) => t,
            None => {
                println!("SKIP: No CUDA device");
                return;
            }
        };
    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 4, 2)
            .expect("encode_batch_from_gpu_ptr_f32 (Float64 engine)")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 2, 4) };
}

#[test]
fn test_encode_batch_from_gpu_ptr_f32_zero_samples() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let result = unsafe { engine.encode_batch_from_gpu_ptr_f32(std::ptr::null(), 0, 4, 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("zero") || msg.contains("samples")),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_f32_null_pointer() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let result = unsafe { engine.encode_batch_from_gpu_ptr_f32(std::ptr::null(), 2, 4, 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("null")),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_f32_sample_size_exceeds_state_len() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[1.0; 10]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let result = unsafe {
        engine.encode_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 5, 2)
    };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => {
            assert!(msg.contains("exceeds") || msg.contains("state vector"));
        }
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_batch_from_gpu_ptr_f32_odd_sample_size_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let num_samples = 2;
    let sample_size = 3;
    let num_qubits = 2;
    let (_device, input_d) = match common::copy_f32_to_device(&[1.0, 2.0, 2.0, 2.0, 1.0, 2.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let dlpack_ptr = unsafe {
        engine
            .encode_batch_from_gpu_ptr_f32(
                *input_d.device_ptr() as *const f32,
                num_samples,
                sample_size,
                num_qubits,
            )
            .expect("encode_batch_from_gpu_ptr_f32 odd sample size")
    };
    unsafe {
        common::assert_dlpack_shape_2d_and_delete(
            dlpack_ptr,
            num_samples as i64,
            (1 << num_qubits) as i64,
        )
    };
}

#[test]
fn test_encode_angle_batch_from_gpu_ptr_f32_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let num_samples = 2;
    let num_qubits = 3;
    let (_device, input_d) = match common::copy_f32_to_device(&[
        0.0,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::FRAC_PI_4,
        0.2,
        0.4,
        0.6,
    ]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let dlpack_ptr = unsafe {
        engine
            .encode_angle_batch_from_gpu_ptr_f32(
                *input_d.device_ptr() as *const f32,
                num_samples,
                num_qubits,
                num_qubits,
            )
            .expect("encode_angle_batch_from_gpu_ptr_f32")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, num_samples as i64, 8) };
}

#[test]
fn test_encode_angle_batch_from_gpu_ptr_f32_with_stream_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (device, input_d) = match common::copy_f32_to_device(&[
        0.0_f32,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::FRAC_PI_4,
        0.2_f32,
        0.4_f32,
        0.6_f32,
    ]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let stream = device.fork_default_stream().expect("fork_default_stream");
    let dlpack_ptr = unsafe {
        engine
            .encode_angle_batch_from_gpu_ptr_f32_with_stream(
                *input_d.device_ptr() as *const f32,
                2,
                3,
                3,
                stream.stream as *mut c_void,
            )
            .expect("encode_angle_batch_from_gpu_ptr_f32_with_stream")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 2, 8) };
}

#[test]
fn test_encode_angle_batch_from_gpu_ptr_f32_null_pointer() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let result = unsafe { engine.encode_angle_batch_from_gpu_ptr_f32(std::ptr::null(), 2, 2, 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("null")),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_angle_batch_from_gpu_ptr_f32_sample_size_mismatch() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0_f32, 0.1, 0.2, 0.3, 0.4, 0.5]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let result = unsafe {
        engine.encode_angle_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 2, 3)
    };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => {
            assert!(
                msg.contains("sample_size=3") || msg.contains("got 2"),
                "msg: {msg}"
            );
        }
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_angle_batch_from_gpu_ptr_f32_zero_samples() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let result = unsafe { engine.encode_angle_batch_from_gpu_ptr_f32(std::ptr::null(), 0, 2, 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("zero") || msg.contains("samples")),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_angle_batch_from_gpu_ptr_f32_non_finite_rejected() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) =
        match common::copy_f32_to_device(&[0.0_f32, f32::NAN, 0.2_f32, 0.3_f32]) {
            Some(t) => t,
            None => {
                println!("SKIP: No CUDA device");
                return;
            }
        };
    let result = unsafe {
        engine.encode_angle_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 2, 2)
    };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => {
            assert!(
                msg.contains("non-finite") || msg.contains("NaN"),
                "msg: {msg}"
            );
        }
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_angle_batch_from_gpu_ptr_f32_infinity_rejected() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) =
        match common::copy_f32_to_device(&[0.0_f32, f32::INFINITY, 0.2_f32, 0.3_f32]) {
            Some(t) => t,
            None => {
                println!("SKIP: No CUDA device");
                return;
            }
        };
    let result = unsafe {
        engine.encode_angle_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 2, 2)
    };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => {
            assert!(
                msg.contains("non-finite") || msg.contains("Inf"),
                "msg: {msg}"
            );
        }
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_angle_batch_from_gpu_ptr_f32_success_f64_engine() {
    let Some(engine) = common::qdp_engine_with_precision(Precision::Float64) else {
        println!("SKIP: No GPU");
        return;
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[
        0.0_f32,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::FRAC_PI_4,
        0.2_f32,
        0.4_f32,
        0.6_f32,
    ]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let dlpack_ptr = unsafe {
        engine
            .encode_angle_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 3, 3)
            .expect("encode_angle_batch_from_gpu_ptr_f32 (Float64 engine)")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 2, 8) };
}

// ── Basis f32 batch from GPU pointer ────────────────────────────────────

#[test]
fn test_encode_basis_batch_from_gpu_ptr_f32_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let num_samples = 3;
    let num_qubits = 3;
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0_f32, 3.0_f32, 7.0_f32]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let dlpack_ptr = unsafe {
        engine
            .encode_basis_batch_from_gpu_ptr_f32(
                *input_d.device_ptr() as *const f32,
                num_samples,
                1,
                num_qubits,
            )
            .expect("encode_basis_batch_from_gpu_ptr_f32")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, num_samples as i64, 8) };
}

#[test]
fn test_encode_basis_batch_from_gpu_ptr_f32_with_stream_success() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (device, input_d) = match common::copy_f32_to_device(&[1.0_f32, 2.0_f32]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let stream = device.fork_default_stream().expect("fork_default_stream");
    let dlpack_ptr = unsafe {
        engine
            .encode_basis_batch_from_gpu_ptr_f32_with_stream(
                *input_d.device_ptr() as *const f32,
                2,
                1,
                2,
                stream.stream as *mut c_void,
            )
            .expect("encode_basis_batch_from_gpu_ptr_f32_with_stream")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 2, 4) };
}

#[test]
fn test_encode_basis_batch_from_gpu_ptr_f32_null_pointer() {
    let Some(engine) = engine_f32() else {
        println!("SKIP: No GPU");
        return;
    };
    let result = unsafe { engine.encode_basis_batch_from_gpu_ptr_f32(std::ptr::null(), 2, 1, 2) };
    assert!(result.is_err());
}

#[test]
fn test_encode_basis_batch_from_gpu_ptr_f32_zero_samples() {
    let Some(engine) = engine_f32() else {
        println!("SKIP: No GPU");
        return;
    };
    let result = unsafe { engine.encode_basis_batch_from_gpu_ptr_f32(std::ptr::null(), 0, 1, 2) };
    assert!(result.is_err());
    match result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("samples"), "msg: {msg}"),
        e => panic!("expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_basis_batch_from_gpu_ptr_f32_sample_size_mismatch() {
    let Some(engine) = engine_f32() else {
        println!("SKIP: No GPU");
        return;
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0_f32, 1.0_f32]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let result = unsafe {
        engine.encode_basis_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 2, 2)
    };
    assert!(result.is_err());
    match result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("sample_size=1"), "msg: {msg}"),
        e => panic!("expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_basis_batch_from_gpu_ptr_f32_non_finite_rejected() {
    let Some(engine) = engine_f32() else {
        println!("SKIP: No GPU");
        return;
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0_f32, f32::NAN]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let result = unsafe {
        engine.encode_basis_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 1, 2)
    };
    assert!(result.is_err());
    match result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("non-finite"), "msg: {msg}"),
        e => panic!("expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_basis_batch_from_gpu_ptr_f32_out_of_range_rejected() {
    let Some(engine) = engine_f32() else {
        println!("SKIP: No GPU");
        return;
    };
    // state_len = 2^2 = 4; index 10 is out of range
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0_f32, 10.0_f32]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let result = unsafe {
        engine.encode_basis_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 1, 2)
    };
    assert!(result.is_err());
    match result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("out of range"), "msg: {msg}"),
        e => panic!("expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_basis_batch_from_gpu_ptr_f32_non_integer_rejected() {
    let Some(engine) = engine_f32() else {
        println!("SKIP: No GPU");
        return;
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0_f32, 1.5_f32]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let result = unsafe {
        engine.encode_basis_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 1, 2)
    };
    assert!(result.is_err());
    match result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("non-integer"), "msg: {msg}"),
        e => panic!("expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_basis_batch_from_gpu_ptr_f32_negative_rejected() {
    let Some(engine) = engine_f32() else {
        println!("SKIP: No GPU");
        return;
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[0.0_f32, -1.0_f32]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let result = unsafe {
        engine.encode_basis_batch_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 2, 1, 2)
    };
    assert!(result.is_err());
    match result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("negative"), "msg: {msg}"),
        e => panic!("expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_basis_from_gpu_ptr_f32_single_sample_success() {
    let Some(engine) = engine_f32() else {
        println!("SKIP: No GPU");
        return;
    };
    let (_device, input_d) = match common::copy_f32_to_device(&[5.0_f32]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let dlpack_ptr = unsafe {
        engine
            .encode_basis_from_gpu_ptr_f32(*input_d.device_ptr() as *const f32, 3)
            .expect("encode_basis_from_gpu_ptr_f32")
    };
    unsafe { common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, 8) };
}
