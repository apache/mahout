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

use qdp_core::io::{read_arrow_ipc_batch, read_parquet_batch};
use arrow::array::{Float64Array, FixedSizeListArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter as ArrowFileWriter;
use std::fs::{self, File};
use std::sync::Arc;

mod common;

#[test]
fn test_read_arrow_ipc_fixed_size_list() {
    let temp_path = "/tmp/test_arrow_ipc_fixed.arrow";
    let num_samples = 10;
    let sample_size = 16;

    // Create test data
    let mut all_values = Vec::new();
    for i in 0..num_samples {
        for j in 0..sample_size {
            all_values.push((i * sample_size + j) as f64);
        }
    }

    // Write Arrow IPC with FixedSizeList format
    let values_array = Float64Array::from(all_values.clone());
    let field = Arc::new(Field::new("item", DataType::Float64, false));
    let list_array = FixedSizeListArray::new(
        field,
        sample_size as i32,
        Arc::new(values_array),
        None,
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float64, false)),
            sample_size as i32,
        ),
        false,
    )]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(list_array)],
    )
    .unwrap();

    let file = File::create(temp_path).unwrap();
    let mut writer = ArrowFileWriter::try_new(file, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();

    // Read and verify
    let (data, samples, size) = read_arrow_ipc_batch(temp_path).unwrap();

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
fn test_read_arrow_ipc_list() {
    let temp_path = "/tmp/test_arrow_ipc_list.arrow";
    let num_samples = 5;
    let sample_size = 8;

    // Create test data with List format
    let mut list_builder = arrow::array::ListBuilder::new(Float64Array::builder(num_samples * sample_size));

    for i in 0..num_samples {
        let values: Vec<f64> = (0..sample_size).map(|j| (i * sample_size + j) as f64).collect();
        list_builder.values().append_slice(&values);
        list_builder.append(true);
    }

    let list_array = list_builder.finish();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        false,
    )]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(list_array)],
    )
    .unwrap();

    let file = File::create(temp_path).unwrap();
    let mut writer = ArrowFileWriter::try_new(file, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();

    // Read and verify
    let (data, samples, size) = read_arrow_ipc_batch(temp_path).unwrap();

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
fn test_arrow_ipc_inconsistent_sizes_fails() {
    let temp_path = "/tmp/test_arrow_ipc_inconsistent.arrow";

    // Create data with inconsistent sample sizes
    let mut list_builder = arrow::array::ListBuilder::new(Float64Array::builder(20));

    // First sample: 4 elements
    list_builder.values().append_slice(&[1.0, 2.0, 3.0, 4.0]);
    list_builder.append(true);

    // Second sample: 8 elements (inconsistent!)
    list_builder.values().append_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    list_builder.append(true);

    let list_array = list_builder.finish();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        false,
    )]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(list_array)],
    )
    .unwrap();

    let file = File::create(temp_path).unwrap();
    let mut writer = ArrowFileWriter::try_new(file, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();

    // Should fail due to inconsistent sizes
    let result = read_arrow_ipc_batch(temp_path);
    assert!(result.is_err());

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_arrow_ipc_empty_file_fails() {
    let result = read_arrow_ipc_batch("/tmp/nonexistent_arrow_file_12345.arrow");
    assert!(result.is_err());
}

#[test]
fn test_arrow_ipc_large_batch() {
    let temp_path = "/tmp/test_arrow_ipc_large.arrow";
    let num_samples = 100;
    let sample_size = 64;

    // Create large dataset
    let mut all_values = Vec::with_capacity(num_samples * sample_size);
    for i in 0..num_samples {
        for j in 0..sample_size {
            all_values.push((i * sample_size + j) as f64 / (num_samples * sample_size) as f64);
        }
    }

    // Write as FixedSizeList
    let values_array = Float64Array::from(all_values.clone());
    let field = Arc::new(Field::new("item", DataType::Float64, false));
    let list_array = FixedSizeListArray::new(
        field,
        sample_size as i32,
        Arc::new(values_array),
        None,
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float64, false)),
            sample_size as i32,
        ),
        false,
    )]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(list_array)],
    )
    .unwrap();

    let file = File::create(temp_path).unwrap();
    let mut writer = ArrowFileWriter::try_new(file, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();

    // Read and verify
    let (data, samples, size) = read_arrow_ipc_batch(temp_path).unwrap();

    assert_eq!(samples, num_samples);
    assert_eq!(size, sample_size);
    assert_eq!(data.len(), all_values.len());

    for i in 0..data.len() {
        assert!((data[i] - all_values[i]).abs() < 1e-10);
    }

    // Cleanup
    fs::remove_file(temp_path).unwrap();
}
