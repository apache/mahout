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

//! Direct tests for ParquetReader and ParquetStreamingReader implementations.
//!
//! These tests exercise the reader types directly, providing coverage for
//! `qdp/qdp-core/src/readers/parquet.rs`.

use arrow::array::{Array, ArrayRef, Float64Array, ListArray, FixedSizeListArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use qdp_core::readers::{ParquetReader, ParquetStreamingReader};
use qdp_core::reader::{DataReader, StreamingDataReader, NullHandling};
use std::fs::File;
use std::sync::Arc;

/// Helper: Create a List<Float64> column from a vector of vectors
fn create_list_array(data: Vec<Vec<f64>>) -> ListArray {
    let values = Float64Array::from_iter_values(data.iter().flatten().copied());
    let offsets: Vec<i32> = data
        .iter()
        .scan(0, |acc, v| {
            let current = *acc;
            *acc += v.len() as i32;
            Some(current)
        })
        .chain(std::iter::once(data.iter().map(|v| v.len() as i32).sum()))
        .collect();
    
    let offsets = OffsetBuffer::new(offsets.into());
    let field = Arc::new(Field::new_list_field(DataType::Float64, false));
    ListArray::new(field, offsets, Arc::new(values), None)
}

/// Helper: Create a FixedSizeList<Float64> column from a vector of vectors
fn create_fixed_list_array(data: Vec<Vec<f64>>, list_size: i32) -> FixedSizeListArray {
    let values = Float64Array::from_iter_values(data.iter().flatten().copied());
    let field = Arc::new(Field::new("item", DataType::Float64, false));
    FixedSizeListArray::new(field, list_size, Arc::new(values), None)
}

/// Helper: Write a List<Float64> Parquet file for testing
fn write_list_parquet(path: &str, data: Vec<Vec<f64>>) {
    let list_array = create_list_array(data);
    let field = Arc::new(Field::new("data", list_array.data_type().clone(), false));
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![Arc::new(list_array)]).unwrap();
    
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Helper: Write a FixedSizeList<Float64> Parquet file for testing
fn write_fixed_list_parquet(path: &str, data: Vec<Vec<f64>>, list_size: i32) {
    let list_array = create_fixed_list_array(data, list_size);
    let field = Arc::new(Field::new("data", list_array.data_type().clone(), false));
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![Arc::new(list_array)]).unwrap();
    
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Helper: Write a scalar Float64 Parquet file for basis encoding tests
fn write_scalar_parquet(path: &str, data: Vec<f64>) {
    let array = Float64Array::from_iter_values(data);
    let field = Arc::new(Field::new("data", DataType::Float64, false));
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
    
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn test_parquet_reader_rejects_missing_file() {
    let result = ParquetReader::new("/tmp/nonexistent_file_12345.parquet", None, NullHandling::FillZero);
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Expected error"),
    };
    assert!(err_msg.contains("not found") || err_msg.contains("No such file"));
}

#[test]
fn test_parquet_reader_rejects_bad_schema() {
    let temp_path = "/tmp/test_bad_schema.parquet";
    
    // Create a file with a non-List column (just Float64)
    let data = vec![1.0, 2.0, 3.0, 4.0];
    write_scalar_parquet(temp_path, data);
    
    // ParquetReader expects List<Float64> or FixedSizeList<Float64>
    let result = ParquetReader::new(temp_path, None, NullHandling::FillZero);
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Expected error"),
    };
    assert!(err_msg.contains("Expected List<Float64> or FixedSizeList<Float64>"));
    
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_parquet_reader_rejects_multiple_columns() {
    let temp_path = "/tmp/test_multiple_columns.parquet";
    
    // Create a file with two columns
    let list1 = create_list_array(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let list2 = create_list_array(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let field1 = Arc::new(Field::new("col1", list1.data_type().clone(), false));
    let field2 = Arc::new(Field::new("col2", list2.data_type().clone(), false));
    let schema = Arc::new(Schema::new(vec![field1, field2]));
    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![Arc::new(list1), Arc::new(list2)]).unwrap();
    
    let file = File::create(temp_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    
    let result = ParquetReader::new(temp_path, None, NullHandling::FillZero);
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Expected error"),
    };
    assert!(err_msg.contains("Expected exactly one column"));
    
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_parquet_reader_handles_list_float64() {
    let temp_path = "/tmp/test_list_float64.parquet";
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
    
    write_list_parquet(temp_path, test_data.clone());
    
    let mut reader = ParquetReader::new(temp_path, None, NullHandling::FillZero).unwrap();
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();
    
    assert_eq!(num_samples, 3);
    assert_eq!(sample_size, 3);
    assert_eq!(data.len(), 9);
    
    // Verify data is flattened correctly
    let expected: Vec<f64> = test_data.iter().flatten().copied().collect();
    assert_eq!(data, expected);
    
    // Verify metadata accessors
    assert_eq!(reader.get_sample_size(), Some(3));
    assert_eq!(reader.get_num_samples(), Some(3));
    
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_parquet_reader_handles_fixed_list_float64() {
    let temp_path = "/tmp/test_fixed_list_float64.parquet";
    let test_data = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    
    write_fixed_list_parquet(temp_path, test_data.clone(), 4);
    
    let mut reader = ParquetReader::new(temp_path, None, NullHandling::FillZero).unwrap();
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();
    
    assert_eq!(num_samples, 2);
    assert_eq!(sample_size, 4);
    assert_eq!(data.len(), 8);
    
    // Verify data is flattened correctly
    let expected: Vec<f64> = test_data.iter().flatten().copied().collect();
    assert_eq!(data, expected);
    
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_parquet_reader_rejects_inconsistent_sample_sizes() {
    let temp_path = "/tmp/test_inconsistent_sizes.parquet";
    
    // Create a ListArray with inconsistent row sizes
    let values = Float64Array::from_iter_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let offsets = OffsetBuffer::new(vec![0, 2, 5].into()); // Row 0 has 2 elements, Row 1 has 3 elements
    let field = Arc::new(Field::new_list_field(DataType::Float64, false));
    let list_array = ListArray::new(field, offsets, Arc::new(values), None);
    
    let schema_field = Arc::new(Field::new("data", list_array.data_type().clone(), false));
    let schema = Arc::new(Schema::new(vec![schema_field]));
    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![Arc::new(list_array)]).unwrap();
    
    let file = File::create(temp_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    
    let mut reader = ParquetReader::new(temp_path, None, NullHandling::FillZero).unwrap();
    let result = reader.read_batch();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Inconsistent sample sizes"));
    
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_parquet_reader_empty_file_fails() {
    let temp_path = "/tmp/test_empty.parquet";
    
    // Create an empty ListArray
    let values = Float64Array::from_iter_values(Vec::<f64>::new());
    let offsets = OffsetBuffer::new(vec![0].into());
    let field = Arc::new(Field::new_list_field(DataType::Float64, false));
    let list_array = ListArray::new(field, offsets, Arc::new(values), None);
    
    let schema_field = Arc::new(Field::new("data", list_array.data_type().clone(), false));
    let schema = Arc::new(Schema::new(vec![schema_field]));
    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![Arc::new(list_array)]).unwrap();
    
    let file = File::create(temp_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    
    let mut reader = ParquetReader::new(temp_path, None, NullHandling::FillZero).unwrap();
    let result = reader.read_batch();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("no data") || err_msg.contains("contains no data"));
    
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_parquet_streaming_reader_accepts_scalar_float64() {
    let temp_path = "/tmp/test_streaming_scalar.parquet";
    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    write_scalar_parquet(temp_path, test_data.clone());
    
    // ParquetStreamingReader accepts scalar Float64 for basis encoding
    let reader = ParquetStreamingReader::new(temp_path, None, NullHandling::FillZero);
    assert!(reader.is_ok());
    
    let mut reader = reader.unwrap();
    // For scalar data, sample_size should be 1
    assert_eq!(reader.get_sample_size(), None); // Not yet determined until first read
    
    // Read the data
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();
    assert_eq!(num_samples, 5);
    assert_eq!(sample_size, 1);
    assert_eq!(data, test_data);
    
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_parquet_streaming_reader_returns_chunked_data() {
    let temp_path = "/tmp/test_streaming_chunked.parquet";
    let test_data: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64, (i + 1) as f64]).collect();
    
    write_list_parquet(temp_path, test_data.clone());
    
    let mut reader = ParquetStreamingReader::new(temp_path, Some(10), NullHandling::FillZero).unwrap();
    
    // Read in chunks
    let mut all_data = Vec::new();
    let mut total_samples = 0;
    
    loop {
        let mut buffer = vec![0.0; 20]; // Buffer for up to 10 samples of size 2
        let written = reader.read_chunk(&mut buffer).unwrap();
        if written == 0 {
            break;
        }
        all_data.extend_from_slice(&buffer[..written]);
        total_samples += written / 2;
    }
    
    assert_eq!(total_samples, 100);
    assert_eq!(all_data.len(), 200);
    
    // Verify data integrity
    let expected: Vec<f64> = test_data.iter().flatten().copied().collect();
    assert_eq!(all_data, expected);
    
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_parquet_streaming_reader_metadata() {
    let temp_path = "/tmp/test_streaming_metadata.parquet";
    let test_data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    
    write_list_parquet(temp_path, test_data);
    
    let reader = ParquetStreamingReader::new(temp_path, None, NullHandling::FillZero).unwrap();
    
    // Verify metadata before reading
    assert_eq!(reader.total_rows, 2);
    assert_eq!(reader.get_sample_size(), None); // Not determined yet
    
    // After reading, sample_size should be set
    let mut reader = reader;
    let _ = reader.read_batch().unwrap();
    assert_eq!(reader.get_sample_size(), Some(3));
    
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_parquet_reader_double_consumption_fails() {
    let temp_path = "/tmp/test_double_consume.parquet";
    let test_data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    
    write_list_parquet(temp_path, test_data);
    
    let mut reader = ParquetReader::new(temp_path, None, NullHandling::FillZero).unwrap();
    
    // First read should succeed
    let result1 = reader.read_batch();
    assert!(result1.is_ok());
    
    // Second read should fail - reader is consumed
    let result2 = reader.read_batch();
    assert!(result2.is_err());
    let err_msg = result2.unwrap_err().to_string();
    assert!(err_msg.contains("already consumed") || err_msg.contains("consumed"));
    
    std::fs::remove_file(temp_path).unwrap();
}
