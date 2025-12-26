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

// Input validation and error handling tests

use qdp_core::{MahoutError, QdpEngine};

mod common;

#[test]
#[cfg(target_os = "linux")]
fn test_input_validation_invalid_strategy() {
    println!("Testing invalid strategy name rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data = common::create_test_data(100);

    let result = engine.encode(&data, 7, "invalid_strategy");
    assert!(result.is_err(), "Should reject invalid strategy");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("Unknown encoder"),
                "Error message should mention unknown encoder"
            );
            println!("PASS: Correctly rejected invalid strategy: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for invalid strategy"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_input_validation_qubit_mismatch() {
    println!("Testing qubit size validation...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data = common::create_test_data(100);

    // 100 elements need 7 qubits (2^7=128), but we request 6 (2^6=64)
    let result = engine.encode(&data, 6, "amplitude");
    assert!(
        result.is_err(),
        "Should reject data larger than state vector"
    );

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("exceeds state vector size"),
                "Error should mention size mismatch"
            );
            println!("PASS: Correctly rejected qubit mismatch: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for size mismatch"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_input_validation_zero_qubits() {
    println!("Testing zero qubits rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data = common::create_test_data(10);

    let result = engine.encode(&data, 0, "amplitude");
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
fn test_input_validation_max_qubits() {
    println!("Testing maximum qubit limit (30)...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data = common::create_test_data(100);

    let result = engine.encode(&data, 35, "amplitude");
    assert!(result.is_err(), "Should reject excessive qubits");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("exceeds") && msg.contains("30"),
                "Error should mention 30 qubit limit"
            );
            println!("PASS: Correctly rejected excessive qubits: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for max qubits"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_input_validation_batch_zero_samples() {
    println!("Testing zero num_samples rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let batch_data = vec![1.0, 2.0, 3.0, 4.0];
    let result = engine.encode_batch(&batch_data, 0, 4, 2, "amplitude");
    assert!(result.is_err(), "Should reject zero num_samples");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("num_samples must be greater than 0"),
                "Error should mention num_samples requirement"
            );
            println!("PASS: Correctly rejected zero num_samples: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for zero num_samples"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_empty_data() {
    println!("Testing empty data rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data: Vec<f64> = vec![];

    let result = engine.encode(&data, 5, "amplitude");
    assert!(result.is_err(), "Should reject empty data");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("empty"), "Error should mention empty data");
            println!("PASS: Correctly rejected empty data: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for empty data"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_zero_norm_data() {
    println!("Testing zero-norm data rejection...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data = vec![0.0; 128];

    let result = engine.encode(&data, 7, "amplitude");
    assert!(result.is_err(), "Should reject zero-norm data");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("zero norm"), "Error should mention zero norm");
            println!("PASS: Correctly rejected zero-norm data: {}", msg);
        }
        _ => panic!("Expected InvalidInput error for zero norm"),
    }
}

#[test]
fn test_error_types() {
    let err1 = MahoutError::InvalidInput("test".to_string());
    let err2 = MahoutError::Cuda("test cuda error".to_string());

    assert!(format!("{}", err1).contains("Invalid input"));
    assert!(format!("{}", err2).contains("CUDA error"));
}

#[test]
#[cfg(not(target_os = "linux"))]
fn test_non_linux_graceful_failure() {
    let result = QdpEngine::new(0);
    assert!(result.is_err());

    if let Err(e) = result {
        println!("PASS: Non-Linux platform correctly rejected: {}", e);
    }
}
