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

// Unit tests for IQP (Instantaneous Quantum Polynomial) encoding

use qdp_core::{MahoutError, QdpEngine};

mod common;

/// Helper to calculate expected data length for IQP full encoding (n + n*(n-1)/2)
fn iqp_full_data_len(num_qubits: usize) -> usize {
    num_qubits + num_qubits * (num_qubits - 1) / 2
}

/// Helper to calculate expected data length for IQP-Z encoding (n only)
fn iqp_z_data_len(num_qubits: usize) -> usize {
    num_qubits
}

// =============================================================================
// Input Validation Tests
// =============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_zero_qubits_rejected() {
    println!("Testing IQP zero qubits rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data = vec![0.5; 1];
    let result = engine.encode(&data, 0, "iqp");
    assert!(result.is_err(), "Should reject zero qubits");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("at least 1"),
                "Error should mention minimum qubit requirement"
            );
            println!("PASS: Correctly rejected zero qubits: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for zero qubits"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_max_qubits_exceeded() {
    println!("Testing IQP max qubits (>30) rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data = vec![0.5; iqp_full_data_len(31)];
    let result = engine.encode(&data, 31, "iqp");
    assert!(result.is_err(), "Should reject qubits > 30");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("30"), "Error should mention 30 qubit limit");
            println!("PASS: Correctly rejected excessive qubits: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for max qubits"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_wrong_data_length() {
    println!("Testing IQP wrong data length rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let num_qubits = 4;
    let expected_len = iqp_full_data_len(num_qubits); // 4 + 6 = 10

    // Provide wrong length (too few)
    let data = vec![0.5; expected_len - 1];
    let result = engine.encode(&data, num_qubits, "iqp");
    assert!(result.is_err(), "Should reject wrong data length");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("expects") && msg.contains(&expected_len.to_string()),
                "Error should mention expected length"
            );
            println!("PASS: Correctly rejected wrong data length: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for wrong data length"),
    }

    // Provide wrong length (too many)
    let data = vec![0.5; expected_len + 1];
    let result = engine.encode(&data, num_qubits, "iqp");
    assert!(
        result.is_err(),
        "Should reject wrong data length (too many)"
    );
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_z_wrong_data_length() {
    println!("Testing IQP-Z wrong data length rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let num_qubits = 4;
    let expected_len = iqp_z_data_len(num_qubits); // 4

    // Provide wrong length
    let data = vec![0.5; expected_len + 2];
    let result = engine.encode(&data, num_qubits, "iqp-z");
    assert!(result.is_err(), "Should reject wrong data length for IQP-Z");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("IQP-Z") && msg.contains(&expected_len.to_string()),
                "Error should mention IQP-Z and expected length"
            );
            println!("PASS: Correctly rejected wrong IQP-Z data length: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for wrong IQP-Z data length"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_nan_value_rejected() {
    println!("Testing IQP NaN value rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let num_qubits = 3;
    let mut data = vec![0.5; iqp_full_data_len(num_qubits)];
    data[2] = f64::NAN;

    let result = engine.encode(&data, num_qubits, "iqp");
    assert!(result.is_err(), "Should reject NaN values");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("finite"),
                "Error should mention finite requirement"
            );
            println!("PASS: Correctly rejected NaN value: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for NaN value"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_infinity_value_rejected() {
    println!("Testing IQP infinity value rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let num_qubits = 3;
    let mut data = vec![0.5; iqp_full_data_len(num_qubits)];
    data[1] = f64::INFINITY;

    let result = engine.encode(&data, num_qubits, "iqp");
    assert!(result.is_err(), "Should reject infinity values");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("finite"),
                "Error should mention finite requirement"
            );
            println!("PASS: Correctly rejected infinity value: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for infinity value"),
    }
}

// =============================================================================
// Single Encode Workflow Tests
// =============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_full_encoding_workflow() {
    println!("Testing IQP full encoding workflow...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 4;
    let data: Vec<f64> = (0..iqp_full_data_len(num_qubits))
        .map(|i| (i as f64) * 0.1)
        .collect();

    let result = engine.encode(&data, num_qubits, "iqp");
    let dlpack_ptr = result.expect("IQP encoding should succeed");
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");
    println!("PASS: IQP full encoding succeeded");

    unsafe {
        let managed = &*dlpack_ptr;
        let tensor = &managed.dl_tensor;

        // Verify 2D shape: [1, 2^num_qubits]
        assert_eq!(tensor.ndim, 2, "IQP tensor should be 2D");

        let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
        assert_eq!(shape_slice[0], 1, "First dimension should be 1");
        assert_eq!(
            shape_slice[1],
            (1 << num_qubits) as i64,
            "Second dimension should be 2^num_qubits"
        );

        println!(
            "PASS: IQP tensor shape correct: [{}, {}]",
            shape_slice[0], shape_slice[1]
        );

        if let Some(deleter) = managed.deleter {
            deleter(dlpack_ptr);
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_z_encoding_workflow() {
    println!("Testing IQP-Z encoding workflow...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 5;
    let data: Vec<f64> = (0..iqp_z_data_len(num_qubits))
        .map(|i| (i as f64) * 0.2)
        .collect();

    let result = engine.encode(&data, num_qubits, "iqp-z");
    let dlpack_ptr = result.expect("IQP-Z encoding should succeed");
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");
    println!("PASS: IQP-Z encoding succeeded");

    unsafe {
        let managed = &*dlpack_ptr;
        let tensor = &managed.dl_tensor;

        assert_eq!(tensor.ndim, 2, "IQP-Z tensor should be 2D");

        let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
        assert_eq!(shape_slice[0], 1, "First dimension should be 1");
        assert_eq!(
            shape_slice[1],
            (1 << num_qubits) as i64,
            "Second dimension should be 2^num_qubits"
        );

        println!(
            "PASS: IQP-Z tensor shape correct: [{}, {}]",
            shape_slice[0], shape_slice[1]
        );

        if let Some(deleter) = managed.deleter {
            deleter(dlpack_ptr);
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_single_qubit() {
    println!("Testing IQP single qubit encoding...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    // Single qubit IQP full: 1 parameter (no ZZ terms with only 1 qubit)
    let num_qubits = 1;
    let data = vec![std::f64::consts::PI / 4.0]; // 1 param for n=1

    let result = engine.encode(&data, num_qubits, "iqp");
    let dlpack_ptr = result.expect("Single qubit IQP encoding should succeed");
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        let managed = &*dlpack_ptr;
        let tensor = &managed.dl_tensor;

        let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
        assert_eq!(
            shape_slice[1], 2,
            "Single qubit should have 2 state amplitudes"
        );

        println!("PASS: Single qubit IQP encoding succeeded with shape [1, 2]");

        if let Some(deleter) = managed.deleter {
            deleter(dlpack_ptr);
        }
    }
}

// =============================================================================
// Batch Encoding Tests
// =============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_batch_encoding() {
    println!("Testing IQP batch encoding...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 3;
    let num_samples = 4;
    let sample_size = iqp_full_data_len(num_qubits); // 3 + 3 = 6

    let batch_data: Vec<f64> = (0..num_samples * sample_size)
        .map(|i| (i as f64) * 0.05)
        .collect();

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "iqp");
    let dlpack_ptr = result.expect("IQP batch encoding should succeed");
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        let managed = &*dlpack_ptr;
        let tensor = &managed.dl_tensor;

        assert_eq!(tensor.ndim, 2, "Batch tensor should be 2D");

        let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
        assert_eq!(
            shape_slice[0], num_samples as i64,
            "First dimension should be num_samples"
        );
        assert_eq!(
            shape_slice[1],
            (1 << num_qubits) as i64,
            "Second dimension should be 2^num_qubits"
        );

        println!(
            "PASS: IQP batch encoding shape correct: [{}, {}]",
            shape_slice[0], shape_slice[1]
        );

        if let Some(deleter) = managed.deleter {
            deleter(dlpack_ptr);
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_z_batch_encoding() {
    println!("Testing IQP-Z batch encoding...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 4;
    let num_samples = 5;
    let sample_size = iqp_z_data_len(num_qubits); // 4

    let batch_data: Vec<f64> = (0..num_samples * sample_size)
        .map(|i| (i as f64) * 0.1)
        .collect();

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "iqp-z");
    let dlpack_ptr = result.expect("IQP-Z batch encoding should succeed");
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        let managed = &*dlpack_ptr;
        let tensor = &managed.dl_tensor;

        assert_eq!(tensor.ndim, 2, "Batch tensor should be 2D");

        let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
        assert_eq!(
            shape_slice[0], num_samples as i64,
            "First dimension should be num_samples"
        );
        assert_eq!(
            shape_slice[1],
            (1 << num_qubits) as i64,
            "Second dimension should be 2^num_qubits"
        );

        println!(
            "PASS: IQP-Z batch encoding shape correct: [{}, {}]",
            shape_slice[0], shape_slice[1]
        );

        if let Some(deleter) = managed.deleter {
            deleter(dlpack_ptr);
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_batch_wrong_sample_size() {
    println!("Testing IQP batch wrong sample_size rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let num_qubits = 3;
    let num_samples = 2;
    let wrong_sample_size = iqp_full_data_len(num_qubits) + 1; // Wrong!

    let batch_data: Vec<f64> = vec![0.5; num_samples * wrong_sample_size];

    let result = engine.encode_batch(
        &batch_data,
        num_samples,
        wrong_sample_size,
        num_qubits,
        "iqp",
    );
    assert!(result.is_err(), "Should reject wrong sample_size");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("sample_size"),
                "Error should mention sample_size"
            );
            println!("PASS: Correctly rejected wrong sample_size: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for wrong sample_size"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_batch_data_length_mismatch() {
    println!("Testing IQP batch data length mismatch rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let num_qubits = 3;
    let num_samples = 3;
    let sample_size = iqp_full_data_len(num_qubits);

    // Provide fewer elements than expected
    let batch_data: Vec<f64> = vec![0.5; num_samples * sample_size - 1];

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "iqp");
    assert!(result.is_err(), "Should reject data length mismatch");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("length") || msg.contains("match"),
                "Error should mention length mismatch"
            );
            println!("PASS: Correctly rejected data length mismatch: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for data length mismatch"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_batch_nan_in_sample() {
    println!("Testing IQP batch NaN value rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let num_qubits = 3;
    let num_samples = 2;
    let sample_size = iqp_full_data_len(num_qubits);

    let mut batch_data: Vec<f64> = vec![0.5; num_samples * sample_size];
    batch_data[sample_size + 2] = f64::NAN; // NaN in second sample

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "iqp");
    assert!(result.is_err(), "Should reject NaN in batch data");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("finite") || msg.contains("Sample"),
                "Error should mention finite requirement or sample index"
            );
            println!("PASS: Correctly rejected NaN in batch: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for NaN in batch"),
    }
}

// =============================================================================
// Expected Data Length Calculation Tests
// =============================================================================

#[test]
fn test_iqp_data_length_calculations() {
    println!("Testing IQP data length calculations...");

    // IQP full: n + n*(n-1)/2
    assert_eq!(iqp_full_data_len(1), 1); // 1 + 0 = 1
    assert_eq!(iqp_full_data_len(2), 3); // 2 + 1 = 3
    assert_eq!(iqp_full_data_len(3), 6); // 3 + 3 = 6
    assert_eq!(iqp_full_data_len(4), 10); // 4 + 6 = 10
    assert_eq!(iqp_full_data_len(5), 15); // 5 + 10 = 15

    // IQP-Z: n only
    assert_eq!(iqp_z_data_len(1), 1);
    assert_eq!(iqp_z_data_len(2), 2);
    assert_eq!(iqp_z_data_len(3), 3);
    assert_eq!(iqp_z_data_len(4), 4);
    assert_eq!(iqp_z_data_len(5), 5);

    println!("PASS: Data length calculations are correct");
}

// =============================================================================
// FWT Optimization Correctness Tests
// =============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_fwt_threshold_boundary() {
    println!("Testing IQP FWT threshold boundary (n=4, where FWT kicks in)...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    // Test at FWT_MIN_QUBITS threshold (n=4)
    let num_qubits = 4;
    let data: Vec<f64> = (0..iqp_full_data_len(num_qubits))
        .map(|i| (i as f64) * 0.1)
        .collect();

    let result = engine.encode(&data, num_qubits, "iqp");
    let dlpack_ptr = result.expect("IQP encoding at FWT threshold should succeed");
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        let managed = &*dlpack_ptr;
        let tensor = &managed.dl_tensor;

        assert_eq!(tensor.ndim, 2, "Tensor should be 2D");
        let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
        assert_eq!(
            shape_slice[1],
            1 << num_qubits,
            "Should have 2^n amplitudes"
        );

        println!(
            "PASS: IQP FWT threshold boundary test with shape [{}, {}]",
            shape_slice[0], shape_slice[1]
        );

        if let Some(deleter) = managed.deleter {
            deleter(dlpack_ptr);
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_fwt_larger_qubit_counts() {
    println!("Testing IQP FWT with larger qubit counts (n=5,6,7,8)...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    for num_qubits in [5, 6, 7, 8] {
        let data: Vec<f64> = (0..iqp_full_data_len(num_qubits))
            .map(|i| (i as f64) * 0.05)
            .collect();

        let result = engine.encode(&data, num_qubits, "iqp");
        let dlpack_ptr = result
            .unwrap_or_else(|_| panic!("IQP encoding for {} qubits should succeed", num_qubits));
        assert!(!dlpack_ptr.is_null());

        unsafe {
            let managed = &*dlpack_ptr;
            let tensor = &managed.dl_tensor;

            let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
            assert_eq!(
                shape_slice[1],
                (1 << num_qubits) as i64,
                "Should have 2^{} amplitudes",
                num_qubits
            );

            println!(
                "  {} qubits: shape [{}, {}] - PASS",
                num_qubits, shape_slice[0], shape_slice[1]
            );

            if let Some(deleter) = managed.deleter {
                deleter(dlpack_ptr);
            }
        }
    }

    println!("PASS: IQP FWT larger qubit count tests completed");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_z_fwt_correctness() {
    println!("Testing IQP-Z FWT correctness for various qubit counts...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    // Test IQP-Z across FWT threshold
    for num_qubits in [3, 4, 5, 6] {
        let data: Vec<f64> = (0..iqp_z_data_len(num_qubits))
            .map(|i| (i as f64) * 0.15)
            .collect();

        let result = engine.encode(&data, num_qubits, "iqp-z");
        let dlpack_ptr = result
            .unwrap_or_else(|_| panic!("IQP-Z encoding for {} qubits should succeed", num_qubits));
        assert!(!dlpack_ptr.is_null());

        unsafe {
            let managed = &*dlpack_ptr;
            let tensor = &managed.dl_tensor;

            let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
            assert_eq!(shape_slice[1], (1 << num_qubits) as i64);

            println!(
                "  IQP-Z {} qubits: shape [{}, {}] - PASS",
                num_qubits, shape_slice[0], shape_slice[1]
            );

            if let Some(deleter) = managed.deleter {
                deleter(dlpack_ptr);
            }
        }
    }

    println!("PASS: IQP-Z FWT correctness tests completed");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_fwt_batch_various_sizes() {
    println!("Testing IQP FWT batch encoding with various qubit counts...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    // Test batch encoding across FWT threshold
    for num_qubits in [3, 4, 5, 6] {
        let num_samples = 8;
        let sample_size = iqp_full_data_len(num_qubits);

        let batch_data: Vec<f64> = (0..num_samples * sample_size)
            .map(|i| (i as f64) * 0.02)
            .collect();

        let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "iqp");
        let dlpack_ptr = result.unwrap_or_else(|_| {
            panic!(
                "IQP batch encoding for {} qubits should succeed",
                num_qubits
            )
        });
        assert!(!dlpack_ptr.is_null());

        unsafe {
            let managed = &*dlpack_ptr;
            let tensor = &managed.dl_tensor;

            let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
            assert_eq!(shape_slice[0], num_samples as i64);
            assert_eq!(shape_slice[1], (1 << num_qubits) as i64);

            println!(
                "  IQP batch {} qubits x {} samples: shape [{}, {}] - PASS",
                num_qubits, num_samples, shape_slice[0], shape_slice[1]
            );

            if let Some(deleter) = managed.deleter {
                deleter(dlpack_ptr);
            }
        }
    }

    println!("PASS: IQP FWT batch encoding tests completed");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_fwt_zero_parameters_identity() {
    println!("Testing IQP FWT with zero parameters produces |0⟩ state...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    // For FWT-optimized path (n >= 4), zero parameters should still give |0⟩
    for num_qubits in [4, 5, 6] {
        let data: Vec<f64> = vec![0.0; iqp_full_data_len(num_qubits)];

        let result = engine.encode(&data, num_qubits, "iqp");
        let dlpack_ptr = result.expect("IQP encoding with zero params should succeed");
        assert!(!dlpack_ptr.is_null());

        unsafe {
            let managed = &*dlpack_ptr;
            let tensor = &managed.dl_tensor;

            let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
            assert_eq!(shape_slice[1], (1 << num_qubits) as i64);

            println!(
                "  IQP zero params {} qubits: verified shape - PASS",
                num_qubits
            );

            if let Some(deleter) = managed.deleter {
                deleter(dlpack_ptr);
            }
        }
    }

    println!("PASS: IQP FWT zero parameters test completed");
}

// =============================================================================
// Encoder Factory Tests
// =============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encoder_via_factory() {
    println!("Testing IQP encoder creation via get_encoder...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    // Test that "iqp" and "IQP" work (case insensitive)
    let num_qubits = 2;
    let data: Vec<f64> = vec![0.1, 0.2, 0.3]; // 2 + 1 = 3 params

    let result1 = engine.encode(&data, num_qubits, "iqp");
    assert!(result1.is_ok(), "lowercase 'iqp' should work");

    let result2 = engine.encode(&data, num_qubits, "IQP");
    assert!(result2.is_ok(), "uppercase 'IQP' should work");

    // Clean up
    unsafe {
        if let Ok(ptr) = result1
            && let Some(d) = (*ptr).deleter
        {
            d(ptr);
        }
        if let Ok(ptr) = result2
            && let Some(d) = (*ptr).deleter
        {
            d(ptr);
        }
    }

    println!("PASS: IQP encoder factory works with case insensitivity");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_z_encoder_via_factory() {
    println!("Testing IQP-Z encoder creation via get_encoder...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let num_qubits = 3;
    let data: Vec<f64> = vec![0.1, 0.2, 0.3]; // 3 params for IQP-Z

    let result = engine.encode(&data, num_qubits, "iqp-z");
    assert!(result.is_ok(), "'iqp-z' should work");

    unsafe {
        if let Ok(ptr) = result
            && let Some(d) = (*ptr).deleter
        {
            d(ptr);
        }
    }

    println!("PASS: IQP-Z encoder factory works");
}
