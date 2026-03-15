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

//! Tests for streaming angle encoder validation and coverage.
//!
//! These tests exercise the validation branches in `encoding/angle.rs`:
//! - sample_size == 0 rejection
//! - sample_size > STAGE_SIZE_ELEMENTS rejection
//! - sample_size != num_qubits rejection
//! - non-finite angle rejection with sample/angle indices
//! - happy-path encode reaching kernel launch

use qdp_core::MahoutError;

mod common;

// =============================================================================
// Validation Tests
// =============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_zero_sample_size_rejected() {
    println!("Testing angle streaming zero sample_size rejection...");

    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // For angle encoding, sample_size must equal num_qubits.
    // With sample_size=0 and num_qubits=3, we should get a validation error.
    let num_qubits = 3;
    let num_samples = 2;
    let sample_size = 0; // Invalid: must be > 0
    let batch_data: Vec<f64> = vec![];

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle");
    assert!(result.is_err(), "Should reject sample_size == 0");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("zero") || msg.contains("Zero") || msg.contains("0"),
                "Error should mention zero: got {}",
                msg
            );
            println!("PASS: Correctly rejected zero sample_size: {}", msg);
        }
        _ => panic!("Expected InvalidInput for sample_size == 0, got {:?}", result),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_sample_size_exceeds_stage_capacity() {
    println!("Testing angle streaming sample_size > STAGE_SIZE_ELEMENTS rejection...");

    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // STAGE_SIZE_ELEMENTS is typically 512MB / 8 bytes = 67,108,864 elements
    // We use a value much larger than this to trigger the capacity check
    let num_qubits = 3;
    let num_samples = 1;
    let sample_size = 100_000_000; // Exceeds STAGE_SIZE_ELEMENTS

    let batch_data: Vec<f64> = vec![0.5; num_samples * sample_size.min(1000)];

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle");
    assert!(result.is_err(), "Should reject sample_size > STAGE_SIZE_ELEMENTS");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("capacity") || msg.contains("staging") || msg.contains("exceeds"),
                "Error should mention capacity/staging: got {}",
                msg
            );
            println!("PASS: Correctly rejected oversized sample_size: {}", msg);
        }
        _ => panic!(
            "Expected InvalidInput for sample_size > STAGE_SIZE_ELEMENTS, got {:?}",
            result
        ),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_sample_size_qubit_mismatch() {
    println!("Testing angle streaming sample_size != num_qubits rejection...");

    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // Angle encoding requires exactly one angle per qubit
    let num_qubits = 3;
    let num_samples = 2;
    let sample_size = 5; // Should be 3 (one per qubit)
    let batch_data: Vec<f64> = vec![0.5; num_samples * sample_size];

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle");
    assert!(
        result.is_err(),
        "Should reject sample_size != num_qubits"
    );

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("sample_size") && msg.contains("qubit"),
                "Error should mention sample_size and qubit: got {}",
                msg
            );
            println!("PASS: Correctly rejected sample_size/qubit mismatch: {}", msg);
        }
        _ => panic!(
            "Expected InvalidInput for sample_size != num_qubits, got {:?}",
            result
        ),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_nan_rejected_with_indices() {
    println!("Testing angle streaming NaN rejection with sample/angle indices...");

    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let num_qubits = 3;
    let num_samples = 3;
    let sample_size = num_qubits; // One angle per qubit
    let mut batch_data: Vec<f64> = vec![0.5; num_samples * sample_size];

    // Inject NaN at sample 1, angle 2
    let nan_sample_idx = 1;
    let nan_angle_idx = 2;
    batch_data[nan_sample_idx * sample_size + nan_angle_idx] = f64::NAN;

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle");
    assert!(result.is_err(), "Should reject NaN values");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            // Error message should contain sample index and angle index
            assert!(
                msg.contains("finite"),
                "Error should mention 'finite': got {}",
                msg
            );
            // The error should mention the sample and angle indices
            assert!(
                msg.contains(&format!("Sample {}", nan_sample_idx)) ||
                msg.contains(&format!("sample {}", nan_sample_idx)) ||
                msg.contains(&nan_sample_idx.to_string()),
                "Error should mention sample index {}: got {}",
                nan_sample_idx,
                msg
            );
            assert!(
                msg.contains(&format!("angle {}", nan_angle_idx)) ||
                msg.contains(&format!("Angle {}", nan_angle_idx)) ||
                msg.contains(&nan_angle_idx.to_string()),
                "Error should mention angle index {}: got {}",
                nan_angle_idx,
                msg
            );
            println!("PASS: Correctly rejected NaN with indices: {}", msg);
        }
        _ => panic!("Expected InvalidInput for NaN value, got {:?}", result),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_positive_infinity_rejected() {
    println!("Testing angle streaming positive infinity rejection...");

    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let num_qubits = 3;
    let num_samples = 2;
    let sample_size = num_qubits;
    let mut batch_data: Vec<f64> = vec![0.5; num_samples * sample_size];

    // Inject positive infinity at sample 0, angle 1
    batch_data[1] = f64::INFINITY;

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle");
    assert!(result.is_err(), "Should reject infinity values");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("finite") || msg.contains("inf"),
                "Error should mention finite/inf: got {}",
                msg
            );
            println!("PASS: Correctly rejected positive infinity: {}", msg);
        }
        _ => panic!("Expected InvalidInput for infinity value, got {:?}", result),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_negative_infinity_rejected() {
    println!("Testing angle streaming negative infinity rejection...");

    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let num_qubits = 3;
    let num_samples = 2;
    let sample_size = num_qubits;
    let mut batch_data: Vec<f64> = vec![0.5; num_samples * sample_size];

    // Inject negative infinity
    batch_data[2] = f64::NEG_INFINITY;

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle");
    assert!(result.is_err(), "Should reject negative infinity values");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("finite") || msg.contains("inf"),
                "Error should mention finite/inf: got {}",
                msg
            );
            println!("PASS: Correctly rejected negative infinity: {}", msg);
        }
        _ => panic!(
            "Expected InvalidInput for negative infinity value, got {:?}",
            result
        ),
    }
}

// =============================================================================
// Happy Path Tests (kernel launch coverage)
// =============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_happy_path_kernel_launch() {
    println!("Testing angle streaming happy path (kernel launch)...");

    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 3;
    let num_samples = 4;
    let sample_size = num_qubits; // One angle per qubit
    let batch_data: Vec<f64> = (0..num_samples * sample_size)
        .map(|i| (i as f64) * 0.1)
        .collect();

    let dlpack_ptr = engine
        .encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle")
        .expect("encode_batch should succeed for valid angle data");

    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    // Verify we can access the deleter and clean up
    unsafe {
        common::take_deleter_and_delete(dlpack_ptr);
    }

    println!("PASS: Angle streaming happy path with kernel launch works");
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_single_sample() {
    println!("Testing angle streaming with single sample...");

    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 2;
    let num_samples = 1;
    let sample_size = num_qubits;
    let batch_data: Vec<f64> = vec![0.1, 0.2]; // Two angles for two qubits

    let dlpack_ptr = engine
        .encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle")
        .expect("encode_batch should succeed for single sample");

    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        common::take_deleter_and_delete(dlpack_ptr);
    }

    println!("PASS: Angle streaming single sample works");
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_large_batch() {
    println!("Testing angle streaming with larger batch...");

    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 4;
    let num_samples = 16;
    let sample_size = num_qubits;
    let batch_data: Vec<f64> = (0..num_samples * sample_size)
        .map(|i| ((i % 10) as f64) * 0.05)
        .collect();

    let dlpack_ptr = engine
        .encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle")
        .expect("encode_batch should succeed for larger batch");

    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    unsafe {
        common::take_deleter_and_delete(dlpack_ptr);
    }

    println!("PASS: Angle streaming large batch works");
}

// =============================================================================
// Additional Edge Case Tests
// =============================================================================

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_batch_data_length_mismatch() {
    println!("Testing angle streaming batch data length mismatch...");

    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let num_qubits = 3;
    let num_samples = 2;
    let sample_size = num_qubits;
    // Provide wrong data length (too few elements)
    let batch_data: Vec<f64> = vec![0.5; num_samples * sample_size - 1];

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle");
    assert!(result.is_err(), "Should reject mismatched batch data length");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("length") || msg.contains("match"),
                "Error should mention length: got {}",
                msg
            );
            println!("PASS: Correctly rejected data length mismatch: {}", msg);
        }
        _ => panic!("Expected InvalidInput for data length mismatch, got {:?}", result),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_zero_qubits_rejected() {
    println!("Testing angle streaming zero qubits rejection...");

    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let num_qubits = 0;
    let num_samples = 1;
    let sample_size = 0; // Must match num_qubits
    let batch_data: Vec<f64> = vec![];

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle");
    assert!(result.is_err(), "Should reject zero qubits");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("qubit") || msg.contains("at least"),
                "Error should mention qubit requirement: got {}",
                msg
            );
            println!("PASS: Correctly rejected zero qubits: {}", msg);
        }
        _ => panic!("Expected InvalidInput for zero qubits, got {:?}", result),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_angle_streaming_excessive_qubits_rejected() {
    println!("Testing angle streaming excessive qubits (>30) rejection...");

    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let num_qubits = 31; // Exceeds typical max of 30
    let num_samples = 1;
    let sample_size = num_qubits;
    let batch_data: Vec<f64> = vec![0.5; num_samples * sample_size];

    let result = engine.encode_batch(&batch_data, num_samples, sample_size, num_qubits, "angle");
    assert!(result.is_err(), "Should reject excessive qubits");

    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("30") || msg.contains("qubit"),
                "Error should mention 30 qubit limit: got {}",
                msg
            );
            println!("PASS: Correctly rejected excessive qubits: {}", msg);
        }
        _ => panic!("Expected InvalidInput for excessive qubits, got {:?}", result),
    }
}
