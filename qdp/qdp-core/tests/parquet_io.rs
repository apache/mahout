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

use qdp_core::io::{
    read_parquet, read_parquet_to_arrow, write_arrow_to_parquet, write_parquet,
};
use arrow::array::Float64Array;
use std::fs;

mod common;

#[test]
fn test_write_and_read_parquet() {
    let temp_path = "/tmp/test_quantum_data.parquet";
    let data = common::create_test_data(4);

    // Write data
    write_parquet(temp_path, &data, None).unwrap();

    // Read it back
    let read_data = read_parquet(temp_path).unwrap();

    // Verify
    assert_eq!(data.len(), read_data.len());
    for (original, read) in data.iter().zip(read_data.iter()) {
        assert!((original - read).abs() < 1e-10);
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_write_with_custom_column_name() {
    let temp_path = "/tmp/test_custom_column.parquet";
    let data = vec![1.0, 2.0, 3.0, 4.0];

    // Write with custom column name
    write_parquet(temp_path, &data, Some("quantum_state")).unwrap();

    // Read it back (column name doesn't matter for reading)
    let read_data = read_parquet(temp_path).unwrap();

    assert_eq!(data, read_data);

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_write_empty_data_fails() {
    let temp_path = "/tmp/test_empty.parquet";
    let data: Vec<f64> = vec![];

    let result = write_parquet(temp_path, &data, None);
    assert!(result.is_err());
}

#[test]
fn test_read_nonexistent_file_fails() {
    let result = read_parquet("/tmp/nonexistent_file_12345.parquet");
    assert!(result.is_err());
}

#[test]
fn test_arrow_roundtrip() {
    let temp_path = "/tmp/test_arrow_roundtrip.parquet";
    let data = common::create_test_data(8);
    let array = Float64Array::from(data.clone());

    // Write Arrow array
    write_arrow_to_parquet(temp_path, &array, None).unwrap();

    // Read back as Arrow arrays (chunked)
    let read_chunks = read_parquet_to_arrow(temp_path).unwrap();

    // Verify total length
    let total_len: usize = read_chunks.iter().map(|c| c.len()).sum();
    assert_eq!(array.len(), total_len);

    // Verify data integrity
    let mut offset = 0;
    for chunk in &read_chunks {
        for i in 0..chunk.len() {
            assert!((array.value(offset + i) - chunk.value(i)).abs() < 1e-10);
        }
        offset += chunk.len();
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_write_empty_arrow_fails() {
    let temp_path = "/tmp/test_empty_arrow.parquet";
    let array = Float64Array::from(Vec::<f64>::new());

    let result = write_arrow_to_parquet(temp_path, &array, None);
    assert!(result.is_err());
}

#[test]
fn test_large_dataset() {
    let temp_path = "/tmp/test_large_dataset.parquet";
    let size = 1024;
    let data: Vec<f64> = (0..size).map(|i| i as f64 / size as f64).collect();

    // Write
    write_parquet(temp_path, &data, None).unwrap();

    // Read
    let read_data = read_parquet(temp_path).unwrap();

    // Verify size and sample values
    assert_eq!(data.len(), read_data.len());
    assert!((data[0] - read_data[0]).abs() < 1e-10);
    assert!((data[size - 1] - read_data[size - 1]).abs() < 1e-10);

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_chunked_read_api() {
    let temp_path = "/tmp/test_chunked_api.parquet";
    let data = common::create_test_data(16);

    // Write test data
    write_parquet(temp_path, &data, None).unwrap();
    let chunks = read_parquet_to_arrow(temp_path).unwrap();
    assert!(!chunks.is_empty());
    let total_len: usize = chunks.iter().map(|c| c.len()).sum();
    assert_eq!(total_len, data.len());
    for chunk in &chunks {
        let buffer_ptr = chunk.values().as_ptr();
        assert!(!buffer_ptr.is_null());
        assert_eq!(buffer_ptr as usize % std::mem::align_of::<f64>(), 0);

        unsafe {
            let slice = std::slice::from_raw_parts(buffer_ptr, chunk.len());
            for (i, &value) in slice.iter().enumerate() {
                assert_eq!(value, chunk.value(i));
            }
        }
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}
