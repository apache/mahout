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

//! Tests for streaming amplitude encoder coverage.
//!
//! These tests exercise the AmplitudeEncoder through the streaming path
//! to ensure coverage of validation, state initialization, and chunk encoding.

#[cfg(target_os = "linux")]
use qdp_core::MahoutError;
#[cfg(target_os = "linux")]
use qdp_core::encoding::{AmplitudeEncoder, ChunkEncoder, STAGE_SIZE_ELEMENTS};

mod common;

/// Test that validate_sample_size rejects sample_size == 0.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_rejects_zero_sample_size() {
    let encoder = AmplitudeEncoder;
    let result = encoder.validate_sample_size(0);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("cannot be zero"),
        "Expected error about zero sample size, got: {}",
        err
    );
}

/// Test that validate_sample_size rejects sample_size > STAGE_SIZE_ELEMENTS.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_rejects_oversized_sample() {
    let encoder = AmplitudeEncoder;
    let oversized = STAGE_SIZE_ELEMENTS + 1;
    let result = encoder.validate_sample_size(oversized);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("exceeds staging buffer"),
        "Expected error about exceeding staging buffer, got: {}",
        err
    );
}

/// Test that validate_sample_size accepts valid sample sizes.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_accepts_valid_sample_size() {
    let encoder = AmplitudeEncoder;

    // Test various valid sample sizes
    for size in [1, 2, 4, 8, 16, 64, 256, 1024] {
        let result = encoder.validate_sample_size(size);
        assert!(
            result.is_ok(),
            "Expected sample size {} to be valid, got error: {:?}",
            size,
            result
        );
    }
}

/// Test successful init_state allocation for valid chunk size.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_init_state_success() {
    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let encoder = AmplitudeEncoder;
    let sample_size = 8;
    let num_qubits = 3; // 2^3 = 8 state vector length

    let result = encoder.init_state(&engine, sample_size, num_qubits);
    assert!(
        result.is_ok(),
        "Expected init_state to succeed, got error: {:?}",
        result
    );
}

/// Test end-to-end streaming amplitude encoding from Parquet file with List<Float64>.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_streaming_encode_list() {
    use std::fs;
    use std::sync::Arc;
    use arrow::array::{ArrayRef, Float64Array, ListArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let temp_path = "/tmp/test_amplitude_streaming.parquet";

    // Create test data: 4 samples, each with 8 elements (amplitude encoding needs 2^n elements)
    let data: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ];

    // Build ListArray from the nested data
    let values: Vec<f64> = data.iter().flatten().copied().collect();
    let values_array = Float64Array::from(values);

    let mut offsets = vec![0i32];
    for sample in &data {
        offsets.push(offsets.last().unwrap() + sample.len() as i32);
    }
    let offset_buffer = OffsetBuffer::new(offsets.into());

    let list_array = ListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        offset_buffer,
        Arc::new(values_array),
        None,
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        true,
    )]));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list_array) as ArrayRef])
        .expect("Failed to create record batch");

    let file = fs::File::create(temp_path).expect("Failed to create file");
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .expect("Failed to create Parquet writer");

    writer.write(&batch).expect("Failed to write batch");
    writer.close().expect("Failed to close writer");

    // Encode using amplitude encoding (3 qubits = 2^3 = 8 state vector length)
    let result = engine.encode_from_parquet(temp_path, 3, "amplitude");

    assert!(
        result.is_ok(),
        "Expected streaming encode to succeed, got error: {:?}",
        result
    );

    let dlpack_ptr = result.unwrap();
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    // Verify the shape is correct: [num_samples, state_len] = [4, 8]
    unsafe {
        common::assert_dlpack_shape_2d(dlpack_ptr, 4, 8);
        common::take_deleter_and_delete(dlpack_ptr);
    }

    // Cleanup
    fs::remove_file(temp_path).ok();
}

/// Test streaming amplitude encoding with larger batch to exercise chunking path.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_streaming_encode_large_batch() {
    use std::fs;
    use std::sync::Arc;
    use arrow::array::{ArrayRef, Float64Array, ListArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let temp_path = "/tmp/test_amplitude_streaming_large.parquet";

    // Create 16 samples with 16 elements each (4 qubits = 2^4 = 16 state vector length)
    let mut data: Vec<Vec<f64>> = Vec::new();
    for i in 0..16 {
        let mut sample = vec![0.0; 16];
        sample[i] = 1.0;
        data.push(sample);
    }

    // Build ListArray
    let values: Vec<f64> = data.iter().flatten().copied().collect();
    let values_array = Float64Array::from(values);

    let mut offsets = vec![0i32];
    for sample in &data {
        offsets.push(offsets.last().unwrap() + sample.len() as i32);
    }
    let offset_buffer = OffsetBuffer::new(offsets.into());

    let list_array = ListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        offset_buffer,
        Arc::new(values_array),
        None,
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        true,
    )]));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list_array) as ArrayRef])
        .expect("Failed to create record batch");

    let file = fs::File::create(temp_path).expect("Failed to create file");
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .expect("Failed to create Parquet writer");

    writer.write(&batch).expect("Failed to write batch");
    writer.close().expect("Failed to close writer");

    // Encode using amplitude encoding (4 qubits = 2^4 = 16 state vector length)
    let result = engine.encode_from_parquet(temp_path, 4, "amplitude");

    assert!(
        result.is_ok(),
        "Expected streaming encode to succeed, got error: {:?}",
        result
    );

    let dlpack_ptr = result.unwrap();
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    // Verify the shape is correct: [num_samples, state_len] = [16, 16]
    unsafe {
        common::assert_dlpack_shape_2d(dlpack_ptr, 16, 16);
        common::take_deleter_and_delete(dlpack_ptr);
    }

    // Cleanup
    fs::remove_file(temp_path).ok();
}

/// Test error handling for non-existent Parquet file.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_streaming_file_not_found() {
    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let result = engine.encode_from_parquet("/tmp/nonexistent_file_12345.parquet", 3, "amplitude");

    assert!(result.is_err(), "Expected error for non-existent file");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("not found") || err.to_string().contains("No such file"),
        "Expected file not found error, got: {}",
        err
    );
}

/// Test streaming encode with FixedSizeList<Float64> column format.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_streaming_fixed_size_list() {
    use std::fs;
    use std::sync::Arc;
    use arrow::array::{ArrayRef, FixedSizeListArray, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let temp_path = "/tmp/test_amplitude_fixed_list.parquet";

    // Create FixedSizeList<Float64> data: 4 samples, each with 8 elements
    let values: Vec<f64> = (0..32).map(|i| if i % 8 == 0 { 1.0 } else { 0.0 }).collect();
    let values_array = Float64Array::from(values);

    let list_array = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        8,
        Arc::new(values_array),
        None,
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 8),
        true,
    )]));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list_array) as ArrayRef])
        .expect("Failed to create record batch");

    let file = fs::File::create(temp_path).expect("Failed to create file");
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .expect("Failed to create Parquet writer");

    writer.write(&batch).expect("Failed to write batch");
    writer.close().expect("Failed to close writer");

    // Encode using amplitude encoding (3 qubits = 2^3 = 8 state vector length)
    let result = engine.encode_from_parquet(temp_path, 3, "amplitude");

    assert!(
        result.is_ok(),
        "Expected streaming encode to succeed, got error: {:?}",
        result
    );

    let dlpack_ptr = result.unwrap();
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    // Verify the shape is correct: [num_samples, state_len] = [4, 8]
    unsafe {
        common::assert_dlpack_shape_2d(dlpack_ptr, 4, 8);
        common::take_deleter_and_delete(dlpack_ptr);
    }

    // Cleanup
    fs::remove_file(temp_path).ok();
}

/// Test error for invalid column type in Parquet file.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_rejects_invalid_column_type() {
    use std::fs;
    use std::sync::Arc;
    use arrow::array::{ArrayRef, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let temp_path = "/tmp/test_amplitude_invalid.parquet";

    // Create a parquet file with Float64 column (not List<Float64>)
    let data: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let array = Float64Array::from(data);

    let schema = Arc::new(Schema::new(vec![Field::new("data", DataType::Float64, false)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array) as ArrayRef])
        .expect("Failed to create record batch");

    let file = fs::File::create(temp_path).expect("Failed to create file");
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .expect("Failed to create Parquet writer");

    writer.write(&batch).expect("Failed to write batch");
    writer.close().expect("Failed to close writer");

    // Try to encode - should fail because column is Float64, not List<Float64>
    let result = engine.encode_from_parquet(temp_path, 3, "amplitude");

    assert!(result.is_err(), "Expected error for invalid column type");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("Expected List<Float64>") || err.to_string().contains("Expected FixedSizeList<Float64>"),
        "Expected List<Float64> error, got: {}",
        err
    );

    // Cleanup
    fs::remove_file(temp_path).ok();
}

/// Test error for empty sample in Parquet file.
#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encoder_rejects_empty_data() {
    use std::fs;
    use std::sync::Arc;
    use arrow::array::{ArrayRef, Float64Array, ListArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let Some(engine) = common::qdp_engine() else {
        println!("SKIP: No GPU available");
        return;
    };

    let temp_path = "/tmp/test_amplitude_empty.parquet";

    // Create test data with empty samples (this will cause sample_size to be 0)
    let data: Vec<Vec<f64>> = vec![vec![], vec![]];

    let values: Vec<f64> = data.iter().flatten().copied().collect();
    let values_array = Float64Array::from(values);

    let mut offsets = vec![0i32];
    for sample in &data {
        offsets.push(offsets.last().unwrap() + sample.len() as i32);
    }
    let offset_buffer = OffsetBuffer::new(offsets.into());

    let list_array = ListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        offset_buffer,
        Arc::new(values_array),
        None,
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        true,
    )]));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list_array) as ArrayRef])
        .expect("Failed to create record batch");

    let file = fs::File::create(temp_path).expect("Failed to create file");
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .expect("Failed to create Parquet writer");

    writer.write(&batch).expect("Failed to write batch");
    writer.close().expect("Failed to close writer");

    // Encode should fail due to empty samples
    let result = engine.encode_from_parquet(temp_path, 3, "amplitude");

    // Cleanup before assertion
    fs::remove_file(temp_path).ok();

    // The encode should fail because empty samples result in sample_size of 0
    // which is rejected by validate_sample_size
    assert!(result.is_err(), "Expected error for empty sample data");
}
