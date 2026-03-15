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

use qdp_core::reader::DataReader;
use qdp_core::readers::TorchReader;

/// Test that TorchReader::new() fails for a missing file.
/// This test runs regardless of whether the pytorch feature is enabled.
#[test]
fn test_torch_reader_missing_file() {
    let missing_path = "/tmp/nonexistent_torch_file_12345.pt";
    let result = TorchReader::new(missing_path);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not found") || err_msg.contains("NotImplemented"),
        "Expected 'not found' or 'NotImplemented' error, got: {}",
        err_msg
    );
}

/// Test reader state getters before and after reading.
/// This test runs regardless of whether the pytorch feature is enabled.
#[test]
fn test_torch_reader_getters() {
    // For non-pytorch feature, we can only test the initial state
    // since we can't create a valid .pt file without PyTorch
    #[cfg(not(feature = "pytorch"))]
    {
        // Test with a path that exists (any file will do for new())
        let temp_path = "/tmp/test_torch_getters_dummy.txt";
        std::fs::write(temp_path, "dummy").unwrap();

        let reader = TorchReader::new(temp_path).unwrap();

        // Before read_batch(), getters should return None
        assert_eq!(reader.get_sample_size(), None);
        assert_eq!(reader.get_num_samples(), None);

        // read_batch() should fail with NotImplemented when pytorch feature is disabled
        let result = reader.read_batch();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("NotImplemented") || err_msg.contains("pytorch"),
            "Expected NotImplemented error, got: {}",
            err_msg
        );

        std::fs::remove_file(temp_path).unwrap();
    }
}

#[cfg(feature = "pytorch")]
mod pytorch_tests {
    use super::*;
    use qdp_core::io::read_torch_batch;
    use std::fs;
    use tch::Tensor;

    #[test]
    fn test_torch_reader_basic_1d() {
        let temp_path = "/tmp/test_torch_basic_1d.pt";
        let sample_size = 12;
        let data: Vec<f64> = (0..sample_size).map(|i| i as f64).collect();

        let tensor = Tensor::from_slice(&data);
        tensor.save(temp_path).unwrap();

        let mut reader = TorchReader::new(temp_path).unwrap();

        // Test getters before read
        assert_eq!(reader.get_sample_size(), None);
        assert_eq!(reader.get_num_samples(), None);

        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        assert_eq!(read_samples, 1);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data, data);

        // Test getters after read
        assert_eq!(reader.get_sample_size(), Some(sample_size));
        assert_eq!(reader.get_num_samples(), Some(1));

        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_read_torch_batch_function_2d() {
        let temp_path = "/tmp/test_torch_batch_fn.pt";
        let num_samples = 4;
        let sample_size = 3;
        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();

        let tensor = Tensor::from_slice(&data).reshape([num_samples as i64, sample_size as i64]);
        tensor.save(temp_path).unwrap();

        let (read_data, read_samples, read_size) = read_torch_batch(temp_path).unwrap();

        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data, data);

        fs::remove_file(temp_path).unwrap();
    }

    /// Test that read_batch() rejects a second read with "Reader already consumed".
    #[test]
    fn test_torch_reader_double_read_fails() {
        let temp_path = "/tmp/test_torch_double_read.pt";
        let sample_size = 8;
        let data: Vec<f64> = (0..sample_size).map(|i| i as f64).collect();

        let tensor = Tensor::from_slice(&data);
        tensor.save(temp_path).unwrap();

        let mut reader = TorchReader::new(temp_path).unwrap();

        // First read should succeed
        let result = reader.read_batch();
        assert!(result.is_ok());

        // Second read should fail with "Reader already consumed"
        let result = reader.read_batch();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("already consumed"),
            "Expected 'Reader already consumed' error, got: {}",
            err_msg
        );

        fs::remove_file(temp_path).unwrap();
    }

    /// Test TorchReader with 2D tensor and verify all getters.
    #[test]
    fn test_torch_reader_2d_getters() {
        let temp_path = "/tmp/test_torch_2d_getters.pt";
        let num_samples = 5;
        let sample_size = 10;
        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();

        let tensor = Tensor::from_slice(&data).reshape([num_samples as i64, sample_size as i64]);
        tensor.save(temp_path).unwrap();

        let mut reader = TorchReader::new(temp_path).unwrap();

        // Before read: getters return None
        assert_eq!(reader.get_sample_size(), None);
        assert_eq!(reader.get_num_samples(), None);

        // Read the data
        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        // Verify data integrity
        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data, data);

        // After read: getters return the actual values
        assert_eq!(reader.get_sample_size(), Some(sample_size));
        assert_eq!(reader.get_num_samples(), Some(num_samples));

        fs::remove_file(temp_path).unwrap();
    }

    /// Test TorchReader::new() error for missing file (with pytorch feature).
    #[test]
    fn test_torch_reader_new_missing_file_with_pytorch() {
        let missing_path = "/tmp/nonexistent_file_abcdef.pt";
        let result = TorchReader::new(missing_path);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not found"),
            "Expected 'not found' error, got: {}",
            err_msg
        );
    }
}
