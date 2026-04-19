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

// Integration tests for the streaming angle encoder (qdp-core/src/encoding/angle.rs).
// All tests go through engine.encode_from_parquet() to exercise the ChunkEncoder
// pipeline path.  The validate_sample_size(0) and oversized cases live as unit
// tests inside encoding/angle.rs because they cannot be triggered via a Parquet
// file.

#![cfg(target_os = "linux")]

use qdp_core::MahoutError;

mod common;

// ---- validate_sample_size / init_state rejection tests ----

#[test]
fn test_angle_mismatched_sample_vs_qubits_rejected() {
    let Some(engine) = common::qdp_engine() else {
        return; //for no GPU situation
    };

    let num_qubits = 3;
    let wrong_sample_size = 5; // should equal num_qubits
    let data: Vec<f64> = (0..wrong_sample_size).map(|i| i as f64 * 0.1).collect();

    let path = "/tmp/test_angle_mismatch.parquet";
    common::write_fixed_size_list_parquet(path, &data, wrong_sample_size);

    let result = engine.encode_from_parquet(path, num_qubits, "angle");
    let _ = std::fs::remove_file(path);

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("sample_size")
                    && msg.contains(&num_qubits.to_string())
                    && msg.contains(&wrong_sample_size.to_string()),
                "msg: {msg}"
            );
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_angle_nan_rejected() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let num_qubits = 3;
    let mut data = vec![0.5_f64; num_qubits];
    data[1] = f64::NAN;

    let path = "/tmp/test_angle_nan.parquet";
    common::write_fixed_size_list_parquet(path, &data, num_qubits);

    let result = engine.encode_from_parquet(path, num_qubits, "angle");
    let _ = std::fs::remove_file(path);

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("finite"), "msg: {msg}");
            assert!(msg.contains("Sample 0"), "msg: {msg}");
            assert!(msg.contains("angle 1"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_angle_infinity_rejected() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let num_qubits = 3;
    let mut data = vec![0.5_f64; num_qubits];
    data[0] = f64::INFINITY;

    let path = "/tmp/test_angle_infinity.parquet";
    common::write_fixed_size_list_parquet(path, &data, num_qubits);

    let result = engine.encode_from_parquet(path, num_qubits, "angle");
    let _ = std::fs::remove_file(path);

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("finite"), "msg: {msg}");
            assert!(msg.contains("Sample 0"), "msg: {msg}");
            assert!(msg.contains("angle 0"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

// ---- Successful encoding (kernel launch path) ----

/// Regression: streaming Parquet path accepts mixed-case encoding names via `Encoding::from_str_ci`.
#[test]
fn test_angle_parquet_encoding_case_insensitive() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let num_qubits = 2;
    let data: Vec<f64> = vec![0.1, 0.2];
    let path = "/tmp/test_angle_case.parquet";
    common::write_fixed_size_list_parquet(path, &data, num_qubits);

    let dlpack_ptr = engine
        .encode_from_parquet(path, num_qubits, "Angle")
        .expect("mixed-case 'Angle' should match streaming angle encoder");
    let _ = std::fs::remove_file(path);

    unsafe {
        common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 1, (1 << num_qubits) as i64);
    }
}

#[test]
fn test_angle_successful_encoding_from_parquet() {
    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 4;
    let num_samples = 3;
    let data: Vec<f64> = (0..num_samples * num_qubits)
        .map(|i| (i as f64) * std::f64::consts::PI / (num_qubits * num_samples) as f64)
        .collect();

    let path = "/tmp/test_angle_success.parquet";
    common::write_fixed_size_list_parquet(path, &data, num_qubits);

    let dlpack_ptr = engine
        .encode_from_parquet(path, num_qubits, "angle")
        .expect("angle streaming encode should succeed");
    let _ = std::fs::remove_file(path);

    unsafe {
        common::assert_dlpack_shape_2d_and_delete(
            dlpack_ptr,
            num_samples as i64,
            (1 << num_qubits) as i64,
        );
    }
}

#[test]
fn test_angle_batch_f32_success() {
    let Some(engine) = common::qdp_engine_with_precision(qdp_core::Precision::Float32) else {
        println!("SKIP: No GPU available");
        return;
    };

    let num_qubits = 3;
    let num_samples = 2;
    let data = vec![
        0.0_f32,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::FRAC_PI_4,
        0.2_f32,
        0.4_f32,
        0.6_f32,
    ];

    let dlpack_ptr = engine
        .encode_batch_f32(&data, num_samples, num_qubits, num_qubits, "angle")
        .expect("angle batch encode f32 should succeed");

    unsafe {
        common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, num_samples as i64, 8);
    }
}

#[test]
fn test_angle_batch_f32_rejects_sample_size_mismatch() {
    let Some(engine) = common::qdp_engine_with_precision(qdp_core::Precision::Float32) else {
        println!("SKIP: No GPU available");
        return;
    };

    let data = vec![0.1_f32, 0.2, 0.3, 0.4];
    let result = engine.encode_batch_f32(&data, 2, 2, 3, "angle");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("sample_size=3") || msg.contains("got 2"),
                "msg: {msg}"
            );
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_angle_batch_f32_rejects_nan() {
    let Some(engine) = common::qdp_engine_with_precision(qdp_core::Precision::Float32) else {
        println!("SKIP: No GPU available");
        return;
    };

    let data = vec![0.0_f32, f32::NAN, 0.2, 0.3];
    let result = engine.encode_batch_f32(&data, 2, 2, 2, "angle");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("Sample 0"), "msg: {msg}");
            assert!(msg.contains("angle 1"), "msg: {msg}");
            assert!(msg.contains("finite"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_angle_batch_f32_rejects_infinity() {
    let Some(engine) = common::qdp_engine_with_precision(qdp_core::Precision::Float32) else {
        println!("SKIP: No GPU available");
        return;
    };

    let data = vec![0.0_f32, f32::INFINITY, 0.2, 0.3];
    let result = engine.encode_batch_f32(&data, 2, 2, 2, "angle");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("Sample 0"), "msg: {msg}");
            assert!(msg.contains("angle 1"), "msg: {msg}");
            assert!(msg.contains("finite"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_angle_batch_f32_rejects_zero_samples() {
    let Some(engine) = common::qdp_engine_with_precision(qdp_core::Precision::Float32) else {
        println!("SKIP: No GPU available");
        return;
    };

    let result = engine.encode_batch_f32(&[], 0, 2, 2, "angle");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(
                msg.contains("zero") || msg.contains("samples"),
                "msg: {msg}"
            );
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_angle_batch_f32_rejects_length_overflow() {
    let Some(engine) = common::qdp_engine_with_precision(qdp_core::Precision::Float32) else {
        println!("SKIP: No GPU available");
        return;
    };

    let result = engine.encode_batch_f32(&[], usize::MAX, 2, 2, "angle");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("overflow"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}
