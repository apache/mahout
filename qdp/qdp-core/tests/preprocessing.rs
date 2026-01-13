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

use qdp_core::MahoutError;
use qdp_core::preprocessing::Preprocessor;

#[test]
fn test_validate_input_success() {
    let data = vec![1.0, 0.0];
    assert!(Preprocessor::validate_input(&data, 1).is_ok());

    let data = vec![1.0, 0.0, 0.0, 0.0];
    assert!(Preprocessor::validate_input(&data, 2).is_ok());
}

#[test]
fn test_validate_input_zero_qubits() {
    let data = vec![1.0];
    let result = Preprocessor::validate_input(&data, 0);
    assert!(matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains("at least 1")));
}

#[test]
fn test_validate_input_too_many_qubits() {
    let data = vec![1.0];
    let result = Preprocessor::validate_input(&data, 31);
    assert!(
        matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains("exceeds practical limit"))
    );
}

#[test]
fn test_validate_input_empty_data() {
    let data: Vec<f64> = vec![];
    let result = Preprocessor::validate_input(&data, 1);
    assert!(
        matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains("cannot be empty"))
    );
}

#[test]
fn test_validate_input_data_too_large() {
    let data = vec![1.0, 0.0, 0.0];
    let result = Preprocessor::validate_input(&data, 1);
    assert!(
        matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains("exceeds state vector size"))
    );
}

#[test]
fn test_validate_input_allows_partial_state() {
    let data = vec![0.5, -0.5, 0.25];
    assert!(Preprocessor::validate_input(&data, 3).is_ok());
}

#[test]
fn test_validate_input_max_qubits_boundary() {
    let data = vec![1.0];
    assert!(Preprocessor::validate_input(&data, 30).is_ok());
}

#[test]
fn test_calculate_l2_norm_success() {
    let data = vec![3.0, 4.0];
    let norm = Preprocessor::calculate_l2_norm(&data).unwrap();
    assert!((norm - 5.0).abs() < 1e-10);

    let data = vec![1.0, 1.0];
    let norm = Preprocessor::calculate_l2_norm(&data).unwrap();
    assert!((norm - 2.0_f64.sqrt()).abs() < 1e-10);
}

#[test]
fn test_calculate_l2_norm_zero() {
    let data = vec![0.0, 0.0, 0.0];
    let result = Preprocessor::calculate_l2_norm(&data);
    assert!(matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains("zero norm")));
}

#[test]
fn test_calculate_l2_norm_mixed_signs() {
    let data = vec![-3.0, 4.0];
    let norm = Preprocessor::calculate_l2_norm(&data).unwrap();
    assert!((norm - 5.0).abs() < 1e-10);
}

#[test]
fn test_calculate_l2_norm_matches_sequential_sum() {
    let data: Vec<f64> = (1..=1000).map(|v| v as f64).collect();
    let norm_parallel = Preprocessor::calculate_l2_norm(&data).unwrap();

    let norm_sequential = data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm_parallel - norm_sequential).abs() < 1e-10);
}

#[test]
fn test_calculate_l2_norm_invalid_values() {
    let cases = [
        ("NaN", f64::NAN, "NaN"),
        ("+Inf", f64::INFINITY, "Infinity"),
        ("-Inf", f64::NEG_INFINITY, "Infinity"),
    ];

    for (label, bad_value, expected_fragment) in cases {
        let data = vec![1.0, bad_value, 3.0];
        let result = Preprocessor::calculate_l2_norm(&data);
        assert!(
            matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains(expected_fragment)),
            "case {label} did not produce expected error"
        );
    }
}

#[test]
fn test_calculate_batch_l2_norms_invalid_values() {
    let cases = [
        ("NaN", f64::NAN, "NaN"),
        ("+Inf", f64::INFINITY, "Infinity"),
        ("-Inf", f64::NEG_INFINITY, "Infinity"),
    ];

    for (label, bad_value, expected_fragment) in cases {
        let batch_data = vec![1.0, 2.0, bad_value, 4.0];
        let result = Preprocessor::calculate_batch_l2_norms(&batch_data, 2, 2);
        assert!(
            matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains(expected_fragment)),
            "case {label} did not produce expected error"
        );
    }
}

#[test]
fn test_calculate_batch_l2_norms_success() {
    let batch_data = vec![3.0, 4.0, 5.0, 12.0];
    let norms = Preprocessor::calculate_batch_l2_norms(&batch_data, 2, 2).unwrap();
    assert_eq!(norms.len(), 2);
    assert!((norms[0] - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
    assert!((norms[1] - 13.0).abs() < 1e-10); // sqrt(5^2 + 12^2) = 13
}
