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

// API workflow tests: Engine initialization and encoding

use qdp_core::QdpEngine;

mod common;

#[test]
#[cfg(target_os = "linux")]
fn test_engine_initialization() {
    println!("Testing QdpEngine initialization...");

    let engine = QdpEngine::new(0);

    match engine {
        Ok(_) => println!("PASS: Engine initialized successfully"),
        Err(e) => {
            println!("SKIP: CUDA initialization failed (no GPU available): {:?}", e);
            return;
        }
    }

    assert!(engine.is_ok());
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoding_workflow() {
    println!("Testing amplitude encoding workflow...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let data = common::create_test_data(1024);
    println!("Created test data: {} elements", data.len());

    let result = engine.encode(&data, 10, "amplitude");
    let dlpack_ptr = result.expect("Encoding should succeed");
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");
    println!("PASS: Encoding succeeded, DLPack pointer valid");

    // Simulate PyTorch behavior: manually call deleter to free GPU memory
    unsafe {
        let managed = &mut *dlpack_ptr;
        assert!(managed.deleter.is_some(), "Deleter must be present");

        println!("Calling deleter to free GPU memory");
        let deleter = managed.deleter.take().expect("Deleter function pointer is missing!");
        deleter(dlpack_ptr);
        println!("PASS: Memory freed successfully");
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoding_async_pipeline() {
    println!("Testing amplitude encoding async pipeline path...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    // Use 200000 elements to trigger async pipeline path (ASYNC_THRESHOLD = 131072)
    let data = common::create_test_data(200000);
    println!("Created test data: {} elements", data.len());

    let result = engine.encode(&data, 18, "amplitude");
    let dlpack_ptr = result.expect("Encoding should succeed");
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");
    println!("PASS: Encoding succeeded, DLPack pointer valid");

    unsafe {
        let managed = &mut *dlpack_ptr;
        assert!(managed.deleter.is_some(), "Deleter must be present");

        println!("Calling deleter to free GPU memory");
        let deleter = managed.deleter.take().expect("Deleter function pointer is missing!");
        deleter(dlpack_ptr);
        println!("PASS: Memory freed successfully");
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_batch_dlpack_2d_shape() {
    println!("Testing batch DLPack 2D shape...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    // Create batch data: 3 samples, each with 4 elements (2 qubits)
    let num_samples = 3;
    let num_qubits = 2;
    let sample_size = 4;
    let batch_data: Vec<f64> = (0..num_samples * sample_size)
        .map(|i| (i as f64) / 10.0)
        .collect();

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "amplitude");
    let dlpack_ptr = result.expect("Batch encoding should succeed");
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        let managed = &*dlpack_ptr;
        let tensor = &managed.dl_tensor;

        // Verify 2D shape for batch tensor
        assert_eq!(tensor.ndim, 2, "Batch tensor should be 2D");

        let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
        assert_eq!(shape_slice[0], num_samples as i64, "First dimension should be num_samples");
        assert_eq!(shape_slice[1], (1 << num_qubits) as i64, "Second dimension should be 2^num_qubits");

        let strides_slice = std::slice::from_raw_parts(tensor.strides, tensor.ndim as usize);
        let state_len = 1 << num_qubits;
        assert_eq!(strides_slice[0], state_len as i64, "Stride for first dimension should be state_len");
        assert_eq!(strides_slice[1], 1, "Stride for second dimension should be 1");

        println!("PASS: Batch DLPack tensor has correct 2D shape: [{}, {}]", shape_slice[0], shape_slice[1]);
        println!("PASS: Strides are correct: [{}, {}]", strides_slice[0], strides_slice[1]);

        // Free memory
        if let Some(deleter) = managed.deleter {
            deleter(dlpack_ptr);
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_single_encode_dlpack_2d_shape() {
    println!("Testing single encode returns 2D shape...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let data = common::create_test_data(16);
    let result = engine.encode(&data, 4, "amplitude");
    assert!(result.is_ok(), "Encoding should succeed");

    let dlpack_ptr = result.unwrap();
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        let managed = &*dlpack_ptr;
        let tensor = &managed.dl_tensor;

        // Verify 2D shape for single encode: [1, 2^num_qubits]
        assert_eq!(tensor.ndim, 2, "Single encode should be 2D");

        let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
        assert_eq!(shape_slice[0], 1, "First dimension should be 1 for single encode");
        assert_eq!(shape_slice[1], 16, "Second dimension should be [2^4]");

        let strides_slice = std::slice::from_raw_parts(tensor.strides, tensor.ndim as usize);
        assert_eq!(strides_slice[0], 16, "Stride for first dimension should be state_len");
        assert_eq!(strides_slice[1], 1, "Stride for second dimension should be 1");

        println!(
            "PASS: Single encode returns 2D shape: [{}, {}]",
            shape_slice[0], shape_slice[1]
        );

        // Free memory
        if let Some(deleter) = managed.deleter {
            deleter(dlpack_ptr);
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_dlpack_device_id() {
    println!("Testing DLPack device_id propagation...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let data = common::create_test_data(16);
    let result = engine.encode(&data, 4, "amplitude");
    assert!(result.is_ok(), "Encoding should succeed");

    let dlpack_ptr = result.unwrap();
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        let managed = &*dlpack_ptr;
        let tensor = &managed.dl_tensor;

        // Verify device_id is correctly set (0 for device 0)
        assert_eq!(tensor.device.device_id, 0, "device_id should be 0 for device 0");

        // Verify device_type is CUDA (kDLCUDA = 2)
        use qdp_core::dlpack::DLDeviceType;
        match tensor.device.device_type {
            DLDeviceType::kDLCUDA => println!("PASS: Device type is CUDA"),
            _ => panic!("Expected CUDA device type"),
        }

        println!("PASS: DLPack device_id correctly set to {}", tensor.device.device_id);

        // Free memory
        if let Some(deleter) = managed.deleter {
            deleter(dlpack_ptr);
        }
    }
}
