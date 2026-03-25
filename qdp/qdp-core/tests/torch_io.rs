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

use qdp_core::error::MahoutError;
use qdp_core::reader::DataReader;
use qdp_core::readers::TorchReader;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid = std::process::id();
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);

    std::env::temp_dir().join(format!("qdp_torch_{pid}_{nanos}_{counter}.{suffix}"))
}

fn create_temp_file(suffix: &str) -> PathBuf {
    let path = unique_temp_path(suffix);
    fs::write(&path, []).unwrap();
    path
}

#[test]
fn test_torch_reader_new_accepts_existing_file() {
    let path = create_temp_file("pt");

    let reader = TorchReader::new(&path).unwrap();

    assert_eq!(reader.get_num_samples(), None);
    assert_eq!(reader.get_sample_size(), None);

    fs::remove_file(path).unwrap();
}

#[test]
fn test_torch_reader_new_rejects_missing_file() {
    let missing_path = unique_temp_path("pt");

    let err = match TorchReader::new(&missing_path) {
        Ok(_) => panic!("expected missing-file error"),
        Err(err) => err,
    };

    match err {
        MahoutError::Io(message) => {
            assert!(message.contains("PyTorch file not found"));
            assert!(message.contains(&missing_path.display().to_string()));
        }
        other => panic!("expected Io error, got {other:?}"),
    }
}

#[test]
fn test_torch_reader_default_build_tracks_consumed_state() {
    let path = create_temp_file("pt");
    let mut reader = TorchReader::new(&path).unwrap();

    assert_eq!(reader.get_num_samples(), None);
    assert_eq!(reader.get_sample_size(), None);

    let first_err = reader.read_batch().unwrap_err();
    match first_err {
        MahoutError::NotImplemented(message) => {
            assert!(message.contains("PyTorch reader requires the 'pytorch' feature"));
        }
        other => panic!("expected NotImplemented error, got {other:?}"),
    }

    assert_eq!(reader.get_num_samples(), None);
    assert_eq!(reader.get_sample_size(), None);

    let second_err = reader.read_batch().unwrap_err();
    match second_err {
        MahoutError::InvalidInput(message) => {
            assert_eq!(message, "Reader already consumed");
        }
        other => panic!("expected InvalidInput error, got {other:?}"),
    }

    fs::remove_file(path).unwrap();
}

#[cfg(feature = "pytorch")]
mod pytorch_tests {
    use super::*;
    use qdp_core::io::read_torch_batch;
    use tch::Tensor;

    #[test]
    fn test_torch_reader_basic_1d() {
        let path = create_temp_file("pt");
        let sample_size = 12;
        let data: Vec<f64> = (0..sample_size).map(|i| i as f64).collect();

        let tensor = Tensor::from_slice(&data);
        tensor.save(&path).unwrap();

        let mut reader = TorchReader::new(&path).unwrap();
        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        assert_eq!(read_samples, 1);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data, data);
        assert_eq!(reader.get_num_samples(), Some(1));
        assert_eq!(reader.get_sample_size(), Some(sample_size));

        let second_err = reader.read_batch().unwrap_err();
        match second_err {
            MahoutError::InvalidInput(message) => {
                assert_eq!(message, "Reader already consumed");
            }
            other => panic!("expected InvalidInput error, got {other:?}"),
        }

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_read_torch_batch_function_2d() {
        let path = create_temp_file("pt");
        let num_samples = 4;
        let sample_size = 3;
        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();

        let tensor = Tensor::from_slice(&data).reshape([num_samples as i64, sample_size as i64]);
        tensor.save(&path).unwrap();

        let (read_data, read_samples, read_size) = read_torch_batch(&path).unwrap();

        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data, data);

        fs::remove_file(path).unwrap();
    }
}
