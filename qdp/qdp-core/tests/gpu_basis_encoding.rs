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

// Integration tests for the streaming basis encoder (qdp-core/src/encoding/basis.rs).
// All tests go through engine.encode_from_parquet() to exercise the ChunkEncoder
// pipeline path.  The needs_staging_copy() unit test lives inside encoding/basis.rs
// because BasisEncoder is pub(crate) and not accessible from here.

#![cfg(target_os = "linux")]

use qdp_core::MahoutError;

mod common;

// ---- validate_sample_size rejection ----

#[test]
fn test_basis_wrong_sample_size_rejected() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // sample_size=2 is invalid for basis encoding (must be 1)
    let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
    let wrong_sample_size = 2;

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    common::write_fixed_size_list_parquet(path, &data, wrong_sample_size);

    let result = engine.encode_from_parquet(path, 2, "basis");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("sample_size=1"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

// ---- encode_chunk: invalid index values ----

#[test]
fn test_basis_nan_rejected() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let data = vec![f64::NAN];
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    common::write_fixed_size_list_parquet(path, &data, 1);

    let result = engine.encode_from_parquet(path, 2, "basis");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("finite"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_basis_infinity_rejected() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let data = vec![f64::INFINITY];
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    common::write_fixed_size_list_parquet(path, &data, 1);

    let result = engine.encode_from_parquet(path, 2, "basis");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("finite"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_basis_negative_rejected() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let data = vec![-1.0_f64];
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    common::write_fixed_size_list_parquet(path, &data, 1);

    let result = engine.encode_from_parquet(path, 2, "basis");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("non-negative"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_basis_fractional_rejected() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    let data = vec![1.5_f64];
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    common::write_fixed_size_list_parquet(path, &data, 1);

    let result = engine.encode_from_parquet(path, 2, "basis");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("integer"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

#[test]
fn test_basis_out_of_range_rejected() {
    let Some(engine) = common::qdp_engine() else {
        return;
    };

    // num_qubits=2 → state_size = 2^2 = 4, so index 4 is out of range
    let data = vec![4.0_f64];
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    common::write_fixed_size_list_parquet(path, &data, 1);

    let result = engine.encode_from_parquet(path, 2, "basis");

    assert!(result.is_err());
    match result {
        Err(MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("exceeds state size"), "msg: {msg}");
        }
        _ => panic!("expected InvalidInput, got {:?}", result),
    }
}

// ---- Successful encoding (kernel launch path) ----

#[test]
fn test_basis_successful_encoding_from_parquet() {
    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    // num_qubits=2 → state_size=4, valid indices are 0..=3
    let num_qubits = 2;
    let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0]; // 4 samples, each with index in [0,3]

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    common::write_fixed_size_list_parquet(path, &data, 1);

    let dlpack_ptr = engine
        .encode_from_parquet(path, num_qubits, "basis")
        .expect("basis streaming encode should succeed");

    // 4 samples, state_size = 2^2 = 4
    unsafe {
        common::assert_dlpack_shape_2d_and_delete(dlpack_ptr, 4, (1 << num_qubits) as i64);
    }
}
