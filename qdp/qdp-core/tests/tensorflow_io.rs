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

use bytes::Bytes;
use qdp_core::io::read_tensorflow_batch;
use qdp_core::reader::DataReader;
use qdp_core::readers::TensorFlowReader;
use std::fs;

mod common;

/// Helper function to create a TensorProto file using prost
/// This creates a minimal TensorProto with tensor_content (preferred path)
fn create_tensorflow_file_tensor_content(
    path: &str,
    data: &[f64],
    shape: &[i64],
) -> Result<(), Box<dyn std::error::Error>> {
    use prost::Message;
    use qdp_core::tf_proto::tensorflow;

    // Convert f64 data to bytes (little-endian)
    let mut tensor_content = Vec::with_capacity(data.len() * 8);
    for &value in data {
        tensor_content.extend_from_slice(&value.to_le_bytes());
    }

    let dims: Vec<tensorflow::Dim> = shape.iter().map(|&size| tensorflow::Dim { size }).collect();

    let tensor_proto = tensorflow::TensorProto {
        dtype: 2, // DT_DOUBLE = 2
        tensor_shape: Some(tensorflow::TensorShapeProto {
            dim: dims,
            unknown_rank: false,
        }),
        tensor_content: tensor_content.into(),
        double_val: vec![],
    };

    let mut buf = Vec::new();
    tensor_proto.encode(&mut buf)?;
    fs::write(path, buf)?;
    Ok(())
}

/// Helper function to create a TensorProto file using double_val (fallback path)
fn create_tensorflow_file_double_val(
    path: &str,
    data: &[f64],
    shape: &[i64],
) -> Result<(), Box<dyn std::error::Error>> {
    use prost::Message;
    use qdp_core::tf_proto::tensorflow;

    let dims: Vec<tensorflow::Dim> = shape.iter().map(|&size| tensorflow::Dim { size }).collect();

    let tensor_proto = tensorflow::TensorProto {
        dtype: 2, // DT_DOUBLE = 2
        tensor_shape: Some(tensorflow::TensorShapeProto {
            dim: dims,
            unknown_rank: false,
        }),
        tensor_content: Bytes::new(),
        double_val: data.to_vec(),
    };

    let mut buf = Vec::new();
    tensor_proto.encode(&mut buf)?;
    fs::write(path, buf)?;
    Ok(())
}

#[test]
fn test_read_tensorflow_2d_tensor_content() {
    let temp_path = "/tmp/test_tensorflow_2d_tc.pb";
    let num_samples = 10;
    let sample_size = 16;

    // Create test data
    let mut data = Vec::new();
    for i in 0..num_samples {
        for j in 0..sample_size {
            data.push((i * sample_size + j) as f64);
        }
    }

    // Create TensorProto file with tensor_content
    create_tensorflow_file_tensor_content(
        temp_path,
        &data,
        &[num_samples as i64, sample_size as i64],
    )
    .unwrap();

    // Read and verify
    let (read_data, samples, size) = read_tensorflow_batch(temp_path).unwrap();

    assert_eq!(samples, num_samples);
    assert_eq!(size, sample_size);
    assert_eq!(read_data.len(), num_samples * sample_size);

    for (i, &val) in read_data.iter().enumerate() {
        assert_eq!(val, i as f64);
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_read_tensorflow_2d_double_val() {
    let temp_path = "/tmp/test_tensorflow_2d_dv.pb";
    let num_samples = 5;
    let sample_size = 8;

    // Create test data
    let mut data = Vec::new();
    for i in 0..num_samples {
        for j in 0..sample_size {
            data.push((i * sample_size + j) as f64);
        }
    }

    // Create TensorProto file with double_val (fallback path)
    create_tensorflow_file_double_val(temp_path, &data, &[num_samples as i64, sample_size as i64])
        .unwrap();

    // Read and verify
    let (read_data, samples, size) = read_tensorflow_batch(temp_path).unwrap();

    assert_eq!(samples, num_samples);
    assert_eq!(size, sample_size);
    assert_eq!(read_data.len(), num_samples * sample_size);

    for (i, &val) in read_data.iter().enumerate() {
        assert_eq!(val, i as f64);
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_read_tensorflow_1d_tensor() {
    let temp_path = "/tmp/test_tensorflow_1d.pb";
    let sample_size = 16;

    // Create test data (1D tensor = single sample)
    let data: Vec<f64> = (0..sample_size).map(|i| i as f64).collect();

    // Create TensorProto file
    create_tensorflow_file_tensor_content(temp_path, &data, &[sample_size as i64]).unwrap();

    // Read and verify
    let (read_data, samples, size) = read_tensorflow_batch(temp_path).unwrap();

    assert_eq!(samples, 1); // 1D tensor is treated as single sample
    assert_eq!(size, sample_size);
    assert_eq!(read_data.len(), sample_size);

    for (i, &val) in read_data.iter().enumerate() {
        assert_eq!(val, i as f64);
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_read_tensorflow_large_batch() {
    let temp_path = "/tmp/test_tensorflow_large.pb";
    let num_samples = 100;
    let sample_size = 64;

    // Create large dataset
    let mut data = Vec::with_capacity(num_samples * sample_size);
    for i in 0..num_samples {
        for j in 0..sample_size {
            data.push((i * sample_size + j) as f64 / (num_samples * sample_size) as f64);
        }
    }

    // Create TensorProto file
    create_tensorflow_file_tensor_content(
        temp_path,
        &data,
        &[num_samples as i64, sample_size as i64],
    )
    .unwrap();

    // Read and verify
    let (read_data, samples, size) = read_tensorflow_batch(temp_path).unwrap();

    assert_eq!(samples, num_samples);
    assert_eq!(size, sample_size);
    assert_eq!(read_data.len(), data.len());

    for i in 0..data.len() {
        assert!((data[i] - read_data[i]).abs() < 1e-10);
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_tensorflow_invalid_dtype() {
    let temp_path = "/tmp/test_tensorflow_invalid_dtype.pb";
    use prost::Message;
    use qdp_core::tf_proto::tensorflow;

    // Create TensorProto with wrong dtype (DT_FLOAT = 1 instead of DT_DOUBLE = 2)
    let tensor_proto = tensorflow::TensorProto {
        dtype: 1, // DT_FLOAT, not DT_DOUBLE
        tensor_shape: Some(tensorflow::TensorShapeProto {
            dim: vec![tensorflow::Dim { size: 4 }],
            unknown_rank: false,
        }),
        tensor_content: Bytes::from(vec![0u8; 32]), // 4 * 8 bytes
        double_val: vec![],
    };

    let mut buf = Vec::new();
    tensor_proto.encode(&mut buf).unwrap();
    fs::write(temp_path, buf).unwrap();

    // Should fail with InvalidInput error
    let result = read_tensorflow_batch(temp_path);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("DT_DOUBLE"));
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_tensorflow_empty_file_fails() {
    let result = read_tensorflow_batch("/tmp/nonexistent_tensorflow_file_12345.pb");
    assert!(result.is_err());
}

#[test]
fn test_tensorflow_3d_shape_fails() {
    let temp_path = "/tmp/test_tensorflow_3d.pb";
    use prost::Message;
    use qdp_core::tf_proto::tensorflow;

    // Create TensorProto with 3D shape (unsupported)
    let tensor_proto = tensorflow::TensorProto {
        dtype: 2, // DT_DOUBLE
        tensor_shape: Some(tensorflow::TensorShapeProto {
            dim: vec![
                tensorflow::Dim { size: 2 },
                tensorflow::Dim { size: 3 },
                tensorflow::Dim { size: 4 },
            ],
            unknown_rank: false,
        }),
        tensor_content: Bytes::from(vec![0u8; 2 * 3 * 4 * 8]),
        double_val: vec![],
    };

    let mut buf = Vec::new();
    tensor_proto.encode(&mut buf).unwrap();
    fs::write(temp_path, buf).unwrap();

    // Should fail with InvalidInput error
    let result = read_tensorflow_batch(temp_path);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("Unsupported tensor rank"));
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_tensorflow_reader_direct() {
    let temp_path = "/tmp/test_tensorflow_reader_direct.pb";
    let num_samples = 3;
    let sample_size = 4;

    let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();
    create_tensorflow_file_tensor_content(
        temp_path,
        &data,
        &[num_samples as i64, sample_size as i64],
    )
    .unwrap();

    // Test direct reader usage
    let mut reader = TensorFlowReader::new(temp_path).unwrap();
    assert_eq!(reader.get_num_samples(), Some(num_samples));
    assert_eq!(reader.get_sample_size(), Some(sample_size));

    let (read_data, samples, size) = reader.read_batch().unwrap();
    assert_eq!(samples, num_samples);
    assert_eq!(size, sample_size);
    assert_eq!(read_data, data);

    // Reader should be consumed
    assert!(reader.read_batch().is_err());

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_tensorflow_size_mismatch_fails() {
    let temp_path = "/tmp/test_tensorflow_size_mismatch.pb";
    use prost::Message;
    use qdp_core::tf_proto::tensorflow;

    // Create TensorProto with shape [2, 4] but wrong data size
    let tensor_proto = tensorflow::TensorProto {
        dtype: 2, // DT_DOUBLE
        tensor_shape: Some(tensorflow::TensorShapeProto {
            dim: vec![tensorflow::Dim { size: 2 }, tensorflow::Dim { size: 4 }],
            unknown_rank: false,
        }),
        tensor_content: Bytes::from(vec![0u8; 16]), // Only 16 bytes, should be 2*4*8 = 64 bytes
        double_val: vec![],
    };

    let mut buf = Vec::new();
    tensor_proto.encode(&mut buf).unwrap();
    fs::write(temp_path, buf).unwrap();

    // Should fail with size mismatch error
    let result = read_tensorflow_batch(temp_path);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("size mismatch"));
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}
