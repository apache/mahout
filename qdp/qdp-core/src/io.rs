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

//! I/O module for reading and writing quantum data using Apache Arrow and Parquet.
//!
//! This module provides efficient columnar data exchange with the data science ecosystem,
//! enabling zero-copy interoperability with pandas, polars, and other Arrow-compatible tools.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::error::{MahoutError, Result};

/// Convert Arrow Float64Array to Vec<f64>
///
/// Uses Arrow's internal buffer directly if no nulls, otherwise copies
pub fn arrow_to_vec(array: &Float64Array) -> Vec<f64> {
    if array.null_count() == 0 {
        array.values().to_vec()
    } else {
        array.iter().map(|opt| opt.unwrap_or(0.0)).collect()
    }
}

/// Reads quantum data from a Parquet file.
///
/// Expects a single column named "data" containing Float64 values.
///
/// # Arguments
/// * `path` - Path to the Parquet file
///
/// # Returns
/// Vector of f64 values from the first column
///
/// # Example
/// ```no_run
/// use qdp_core::io::read_parquet;
///
/// let data = read_parquet("quantum_data.parquet").unwrap();
/// ```
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Vec<f64>> {
    let file = File::open(path.as_ref()).map_err(|e| {
        MahoutError::Io(format!("Failed to open Parquet file: {}", e))
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet reader: {}", e))
    })?;

    let metadata = builder.metadata().clone();
    let num_rows = metadata.file_metadata().num_rows() as usize;

    let mut reader = builder.build().map_err(|e| {
        MahoutError::Io(format!("Failed to build Parquet reader: {}", e))
    })?;

    // Pre-allocate with exact capacity to avoid reallocations
    let mut all_data = Vec::with_capacity(num_rows);

    // Read all batches
    while let Some(batch_result) = reader.next() {
        let batch = batch_result.map_err(|e| {
            MahoutError::Io(format!("Failed to read Parquet batch: {}", e))
        })?;

        if batch.num_columns() == 0 {
            return Err(MahoutError::Io(
                "Parquet file has no columns".to_string(),
            ));
        }

        // Extract data from first column
        let column = batch.column(0);
        let float_array = column
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                MahoutError::Io(format!(
                    "Expected Float64 column, got {:?}",
                    column.data_type()
                ))
            })?;

        all_data.extend(float_array.values().iter().copied());
    }

    if all_data.is_empty() {
        return Err(MahoutError::Io(
            "Parquet file contains no data".to_string(),
        ));
    }

    Ok(all_data)
}

/// Writes quantum data to a Parquet file.
///
/// Creates a single column named "data" containing Float64 values.
///
/// # Arguments
/// * `path` - Path to write the Parquet file
/// * `data` - Vector of f64 values to write
/// * `column_name` - Optional column name (defaults to "data")
///
/// # Example
/// ```no_run
/// use qdp_core::io::write_parquet;
///
/// let data = vec![0.5, 0.5, 0.5, 0.5];
/// write_parquet("quantum_data.parquet", &data, None).unwrap();
/// ```
pub fn write_parquet<P: AsRef<Path>>(
    path: P,
    data: &[f64],
    column_name: Option<&str>,
) -> Result<()> {
    if data.is_empty() {
        return Err(MahoutError::InvalidInput(
            "Cannot write empty data to Parquet".to_string(),
        ));
    }

    let col_name = column_name.unwrap_or("data");

    // Create Arrow schema
    let schema = Arc::new(Schema::new(vec![Field::new(
        col_name,
        DataType::Float64,
        false,
    )]));

    // Create Float64Array from slice
    let array = Float64Array::from(Vec::from(data));
    let array_ref: ArrayRef = Arc::new(array);

    // Create RecordBatch
    let batch = RecordBatch::try_new(schema.clone(), vec![array_ref]).map_err(|e| {
        MahoutError::Io(format!("Failed to create RecordBatch: {}", e))
    })?;

    // Write to Parquet file
    let file = File::create(path.as_ref()).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet file: {}", e))
    })?;

    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet writer: {}", e))
    })?;

    writer.write(&batch).map_err(|e| {
        MahoutError::Io(format!("Failed to write Parquet batch: {}", e))
    })?;

    writer.close().map_err(|e| {
        MahoutError::Io(format!("Failed to close Parquet writer: {}", e))
    })?;

    Ok(())
}

/// Reads quantum data from a Parquet file and returns it as an Arrow Float64Array.
///
/// Returns the Arrow array from Parquet
///
/// # Arguments
/// * `path` - Path to the Parquet file
///
/// # Returns
/// Float64Array containing the data from the first column
pub fn read_parquet_to_arrow<P: AsRef<Path>>(path: P) -> Result<Float64Array> {
    let data = read_parquet(path)?;
    Ok(Float64Array::from(data))
}

/// Writes an Arrow Float64Array to a Parquet file.
///
/// Zero-copy write from Arrow format to Parquet.
///
/// # Arguments
/// * `path` - Path to write the Parquet file
/// * `array` - Float64Array to write
/// * `column_name` - Optional column name (defaults to "data")
pub fn write_arrow_to_parquet<P: AsRef<Path>>(
    path: P,
    array: &Float64Array,
    column_name: Option<&str>,
) -> Result<()> {
    if array.is_empty() {
        return Err(MahoutError::InvalidInput(
            "Cannot write empty array to Parquet".to_string(),
        ));
    }

    let col_name = column_name.unwrap_or("data");

    // Create Arrow schema
    let schema = Arc::new(Schema::new(vec![Field::new(
        col_name,
        DataType::Float64,
        false,
    )]));

    let array_ref: ArrayRef = Arc::new(array.clone());

    // Create RecordBatch
    let batch = RecordBatch::try_new(schema.clone(), vec![array_ref]).map_err(|e| {
        MahoutError::Io(format!("Failed to create RecordBatch: {}", e))
    })?;

    // Write to Parquet file
    let file = File::create(path.as_ref()).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet file: {}", e))
    })?;

    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet writer: {}", e))
    })?;

    writer.write(&batch).map_err(|e| {
        MahoutError::Io(format!("Failed to write Parquet batch: {}", e))
    })?;

    writer.close().map_err(|e| {
        MahoutError::Io(format!("Failed to close Parquet writer: {}", e))
    })?;

    Ok(())
}
