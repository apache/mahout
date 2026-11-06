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
use qdp_core::MahoutError;
use qdp_core::io::read_numpy_batch;
use qdp_core::reader::DataReader;
use qdp_core::readers::{NpyMemoryLimit, NumpyReader};
use std::fs;
use std::io::Write;

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

/// Write a v1.0 `.npy` file whose header is `dict` followed by `body_len`
/// zero bytes. Lets a test claim any shape without paying for the data.
fn write_crafted_npy(dict: &str, body_len: usize) -> tempfile::NamedTempFile {
    let mut header = dict.as_bytes().to_vec();
    let unpadded = 10 + header.len() + 1;
    header.resize(header.len() + (64 - unpadded % 64) % 64, b' ');
    header.push(b'\n');

    let mut file = tempfile::Builder::new().suffix(".npy").tempfile().unwrap();
    file.write_all(b"\x93NUMPY\x01\x00").unwrap();
    file.write_all(&(header.len() as u16).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&vec![0u8; body_len]).unwrap();
    file.flush().unwrap();
    file
}

const HUGE_HEADER: &str =
    "{'descr': '<f8', 'fortran_order': False, 'shape': (1000000000000, 10), }";

#[test]
fn small_file_passes_explicit_limit() {
    let array = Array2::from_shape_vec((2, 4), (0..8).map(f64::from).collect()).unwrap();
    let file = tempfile::Builder::new().suffix(".npy").tempfile().unwrap();
    ndarray_npy::write_npy(file.path(), &array).unwrap();

    // Exactly the array size (2 * 4 * 8 = 64 bytes) is allowed.
    let mut reader = NumpyReader::new(file.path())
        .unwrap()
        .with_memory_limit(NpyMemoryLimit::MaxBytes(64));
    let (data, n, s) = reader.read_batch().unwrap();
    assert_eq!((n, s), (2, 4));
    assert_eq!(data, (0..8).map(f64::from).collect::<Vec<_>>());

    // One byte less is refused.
    let mut reader = NumpyReader::new(file.path())
        .unwrap()
        .with_memory_limit(NpyMemoryLimit::MaxBytes(63));
    assert!(matches!(
        reader.read_batch(),
        Err(MahoutError::MemoryAllocation(_))
    ));
}

#[test]
fn huge_header_rejected_by_explicit_limit() {
    // Header claims 10^12 x 10 float64 (~72.76 TiB); the file has no body.
    let file = write_crafted_npy(HUGE_HEADER, 0);
    let mut reader = NumpyReader::new(file.path())
        .unwrap()
        .with_memory_limit(NpyMemoryLimit::MaxBytes(16 << 30));
    let err = reader.read_batch().unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, MahoutError::MemoryAllocation(_)), "{err:?}");
    assert!(msg.contains("(1000000000000, 10)"), "{msg}");
    assert!(msg.contains("72.76 TiB (80000000000000 bytes)"), "{msg}");
    assert!(msg.contains("16.00 GiB"), "{msg}");
    assert!(msg.contains("QDP_NPY_MAX_BYTES"), "{msg}");
    assert!(msg.contains("Parquet"), "{msg}");
}

#[cfg(target_os = "linux")]
#[test]
fn huge_header_rejected_by_auto_limit() {
    // No machine has 2 * 72.76 TiB / 0.8 of available RAM, so the dynamic
    // limit (driven by the real MemAvailable) must refuse this.
    let file = write_crafted_npy(HUGE_HEADER, 0);
    let mut reader = NumpyReader::new(file.path())
        .unwrap()
        .with_memory_limit(NpyMemoryLimit::Auto);
    let err = reader.read_batch().unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, MahoutError::MemoryAllocation(_)), "{err:?}");
    assert!(msg.contains("80% of available host memory"), "{msg}");
    assert!(msg.contains("MemAvailable"), "{msg}");
}

#[test]
fn truncated_body_fails_fast_even_with_limit_disabled() {
    // With the guard off, the on-disk size check must still stop the reader
    // before it allocates 72.76 TiB for a file that holds 80 bytes.
    let file = write_crafted_npy(HUGE_HEADER, 80);
    let mut reader = NumpyReader::new(file.path())
        .unwrap()
        .with_memory_limit(NpyMemoryLimit::Disabled);
    let err = reader.read_batch().unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, MahoutError::InvalidInput(_)), "{err:?}");
    assert!(msg.contains("truncated"), "{msg}");
}

#[test]
fn small_truncated_file_reports_truncation() {
    // (2, 3) float64 needs 48 data bytes; give it 8.
    let file = write_crafted_npy(
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }",
        8,
    );
    let err = NumpyReader::new(file.path())
        .unwrap()
        .with_memory_limit(NpyMemoryLimit::Disabled)
        .read_batch()
        .unwrap_err();
    assert!(err.to_string().contains("needs 48 bytes"), "{err}");
}

#[test]
fn overflowing_shape_is_rejected() {
    // 2^40 * 2^40 * 8 does not fit in u64.
    let file = write_crafted_npy(
        "{'descr': '<f8', 'fortran_order': False, 'shape': (1099511627776, 1099511627776), }",
        0,
    );
    let err = NumpyReader::new(file.path())
        .unwrap()
        .with_memory_limit(NpyMemoryLimit::Disabled)
        .read_batch()
        .unwrap_err();
    assert!(matches!(err, MahoutError::InvalidInput(_)), "{err:?}");
    assert!(err.to_string().contains("overflows"), "{err}");
}

#[test]
fn fortran_order_file_passes_limit_check() {
    let array = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    // Column-major copy of the same logical array.
    let array_f = array.t().as_standard_layout().into_owned().reversed_axes();
    assert!(!array_f.is_standard_layout());
    let file = tempfile::Builder::new().suffix(".npy").tempfile().unwrap();
    ndarray_npy::write_npy(file.path(), &array_f).unwrap();

    let (data, n, s) = NumpyReader::new(file.path())
        .unwrap()
        .with_memory_limit(NpyMemoryLimit::MaxBytes(48))
        .read_batch()
        .unwrap();
    assert_eq!((n, s), (3, 2));
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
