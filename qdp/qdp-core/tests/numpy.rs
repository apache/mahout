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

use ndarray::Array2;
use qdp_core::io::read_numpy_batch;
use qdp_core::reader::DataReader;
use qdp_core::readers::NumpyReader;
use std::fs;

#[test]
fn test_read_numpy_batch_function() {
    let temp_path = "/tmp/test_numpy_batch_fn.npy";
    let num_samples = 10;
    let sample_size = 16;

    // Create test data
    let mut all_values = Vec::new();
    for i in 0..num_samples {
        for j in 0..sample_size {
            all_values.push((i * sample_size + j) as f64);
        }
    }

    // Write NumPy file
    let array = Array2::from_shape_vec((num_samples, sample_size), all_values.clone()).unwrap();
    ndarray_npy::write_npy(temp_path, &array).unwrap();

    // Read using the convenience function
    let (data, samples, size) = read_numpy_batch(temp_path).unwrap();

    assert_eq!(samples, num_samples);
    assert_eq!(size, sample_size);
    assert_eq!(data.len(), num_samples * sample_size);

    for (i, &val) in data.iter().enumerate() {
        assert_eq!(val, i as f64);
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_numpy_reader_with_qubits_data() {
    let temp_path = "/tmp/test_numpy_qubits.npy";
    let num_samples = 5;
    let num_qubits = 3;
    let sample_size = 1 << num_qubits; // 2^3 = 8

    // Create normalized quantum state vectors
    let mut all_values = Vec::new();
    for i in 0..num_samples {
        for j in 0..sample_size {
            // Create a simple pattern
            all_values.push((i * sample_size + j) as f64 / (num_samples * sample_size) as f64);
        }
    }

    // Write NumPy file
    let array = Array2::from_shape_vec((num_samples, sample_size), all_values.clone()).unwrap();
    ndarray_npy::write_npy(temp_path, &array).unwrap();

    // Read it back
    let mut reader = NumpyReader::new(temp_path).unwrap();
    let (data, samples, size) = reader.read_batch().unwrap();

    assert_eq!(samples, num_samples);
    assert_eq!(size, sample_size);
    assert_eq!(data.len(), num_samples * sample_size);

    // Verify data integrity
    for (i, &val) in data.iter().enumerate() {
        let expected = i as f64 / (num_samples * sample_size) as f64;
        assert!((val - expected).abs() < 1e-10);
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_numpy_reader_large_batch() {
    let temp_path = "/tmp/test_numpy_large.npy";
    let num_samples = 100;
    let sample_size = 64; // 2^6

    // Create large dataset
    let mut all_values = Vec::with_capacity(num_samples * sample_size);
    for i in 0..num_samples {
        for j in 0..sample_size {
            all_values.push((i * sample_size + j) as f64 / (num_samples * sample_size) as f64);
        }
    }

    // Write NumPy file
    let array = Array2::from_shape_vec((num_samples, sample_size), all_values.clone()).unwrap();
    ndarray_npy::write_npy(temp_path, &array).unwrap();

    // Read and verify
    let (data, samples, size) = read_numpy_batch(temp_path).unwrap();

    assert_eq!(samples, num_samples);
    assert_eq!(size, sample_size);
    assert_eq!(data.len(), all_values.len());

    for i in 0..data.len() {
        assert!((data[i] - all_values[i]).abs() < 1e-10);
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_numpy_reader_single_sample() {
    let temp_path = "/tmp/test_numpy_single.npy";
    let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let array = Array2::from_shape_vec((1, 8), data.clone()).unwrap();
    ndarray_npy::write_npy(temp_path, &array).unwrap();

    let mut reader = NumpyReader::new(temp_path).unwrap();
    let (read_data, samples, size) = reader.read_batch().unwrap();

    assert_eq!(samples, 1);
    assert_eq!(size, 8);
    assert_eq!(read_data, data);

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}
