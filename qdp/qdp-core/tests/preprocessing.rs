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

use qdp_core::preprocessing::Preprocessor;
use qdp_core::MahoutError;

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
    assert!(matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains("exceeds practical limit")));
}

#[test]
fn test_validate_input_empty_data() {
    let data: Vec<f64> = vec![];
    let result = Preprocessor::validate_input(&data, 1);
    assert!(matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains("cannot be empty")));
}

#[test]
fn test_validate_input_data_too_large() {
    let data = vec![1.0, 0.0, 0.0]; // 3 elements
    let result = Preprocessor::validate_input(&data, 1); // max size 2^1 = 2
    assert!(matches!(result, Err(MahoutError::InvalidInput(msg)) if msg.contains("exceeds state vector size")));
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
