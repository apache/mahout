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

//! I/O utilities for reading and writing quantum data.
//!
//! Provides efficient columnar data exchange via Apache Arrow and Parquet formats.
//!
//! # TODO
//! Consider using generic `T: ArrowPrimitiveType` instead of hardcoded `Float64Array`
//! to support both Float32 and Float64 for flexibility in precision vs performance trade-offs.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, FixedSizeListArray, ListArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader as ArrowFileReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::error::{MahoutError, Result};

/// Build Parquet writer properties optimized for fast decode.
fn fast_decode_writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::SNAPPY) // Light-weight codec; switch to UNCOMPRESSED for max decode speed
        .build()
}

/// Converts an Arrow Float64Array to Vec<f64>.
pub fn arrow_to_vec(array: &Float64Array) -> Vec<f64> {
    if array.null_count() == 0 {
        array.values().to_vec()
    } else {
        array.iter().map(|opt| opt.unwrap_or(0.0)).collect()
    }
}

/// Flattens multiple Arrow Float64Arrays into a single Vec<f64>.
pub fn arrow_to_vec_chunked(arrays: &[Float64Array]) -> Vec<f64> {
    let total_len: usize = arrays.iter().map(|a| a.len()).sum();
    let mut result = Vec::with_capacity(total_len);

    for array in arrays {
        if array.null_count() == 0 {
            result.extend_from_slice(array.values());
        } else {
            result.extend(array.iter().map(|opt| opt.unwrap_or(0.0)));
        }
    }

    result
}

/// Reads Float64 data from a Parquet file.
///
/// Expects a single Float64 column. For zero-copy access, use [`read_parquet_to_arrow`].
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Vec<f64>> {
    let chunks = read_parquet_to_arrow(path)?;
    Ok(arrow_to_vec_chunked(&chunks))
}

/// Writes Float64 data to a Parquet file.
///
/// # Arguments
/// * `path` - Output file path
/// * `data` - Data to write
/// * `column_name` - Column name (defaults to "data")
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

    let schema = Arc::new(Schema::new(vec![Field::new(
        col_name,
        DataType::Float64,
        false,
    )]));

    let array = Float64Array::from_iter_values(data.iter().copied());
    let array_ref: ArrayRef = Arc::new(array);

    let batch = RecordBatch::try_new(schema.clone(), vec![array_ref]).map_err(|e| {
        MahoutError::Io(format!("Failed to create RecordBatch: {}", e))
    })?;

    let file = File::create(path.as_ref()).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet file: {}", e))
    })?;

    let props = fast_decode_writer_props();
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

/// Reads a Parquet file as Arrow Float64Arrays.
///
/// Returns one array per row group for zero-copy access.
pub fn read_parquet_to_arrow<P: AsRef<Path>>(path: P) -> Result<Vec<Float64Array>> {
    let file = File::open(path.as_ref()).map_err(|e| {
        MahoutError::Io(format!("Failed to open Parquet file: {}", e))
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet reader: {}", e))
    })?;

    let mut reader = builder.build().map_err(|e| {
        MahoutError::Io(format!("Failed to build Parquet reader: {}", e))
    })?;

    let mut arrays = Vec::new();

    while let Some(batch_result) = reader.next() {
        let batch = batch_result.map_err(|e| {
            MahoutError::Io(format!("Failed to read Parquet batch: {}", e))
        })?;

        if batch.num_columns() == 0 {
            return Err(MahoutError::Io(
                "Parquet file has no columns".to_string(),
            ));
        }

        let column = batch.column(0);
        if !matches!(column.data_type(), DataType::Float64) {
            return Err(MahoutError::Io(format!(
                "Expected Float64 column, got {:?}",
                column.data_type()
            )));
        }

        let float_array = column
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                MahoutError::Io("Failed to downcast to Float64Array".to_string())
            })?
            .clone();

        arrays.push(float_array);
    }

    if arrays.is_empty() {
        return Err(MahoutError::Io(
            "Parquet file contains no data".to_string(),
        ));
    }

    Ok(arrays)
}

/// Writes an Arrow Float64Array to a Parquet file.
///
/// # Arguments
/// * `path` - Output file path
/// * `array` - Array to write
/// * `column_name` - Column name (defaults to "data")
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

    let schema = Arc::new(Schema::new(vec![Field::new(
        col_name,
        DataType::Float64,
        false,
    )]));

    let array_ref: ArrayRef = Arc::new(array.clone());
    let batch = RecordBatch::try_new(schema.clone(), vec![array_ref]).map_err(|e| {
        MahoutError::Io(format!("Failed to create RecordBatch: {}", e))
    })?;

    let file = File::create(path.as_ref()).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet file: {}", e))
    })?;

    let props = fast_decode_writer_props();
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

/// Reads batch data from a Parquet file with `List<Float64>` column format.
///
/// Returns flattened data suitable for batch encoding.
///
/// # Returns
/// Tuple of `(flattened_data, num_samples, sample_size)`
///
/// # TODO
/// Add OOM protection for very large files
pub fn read_parquet_batch<P: AsRef<Path>>(path: P) -> Result<(Vec<f64>, usize, usize)> {
    let file = File::open(path.as_ref()).map_err(|e| {
        MahoutError::Io(format!("Failed to open Parquet file: {}", e))
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet reader: {}", e))
    })?;

    let total_rows = builder.metadata().file_metadata().num_rows() as usize;

    let mut reader = builder.build().map_err(|e| {
        MahoutError::Io(format!("Failed to build Parquet reader: {}", e))
    })?;

    let mut all_data = Vec::new();
    let mut num_samples = 0;
    let mut sample_size = None;

    while let Some(batch_result) = reader.next() {
        let batch = batch_result.map_err(|e| {
            MahoutError::Io(format!("Failed to read Parquet batch: {}", e))
        })?;

        if batch.num_columns() == 0 {
            return Err(MahoutError::Io("Parquet file has no columns".to_string()));
        }

        let column = batch.column(0);

        if let DataType::List(_) = column.data_type() {
            let list_array = column
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| MahoutError::Io("Failed to downcast to ListArray".to_string()))?;

            for i in 0..list_array.len() {
                let value_array = list_array.value(i);
                let float_array = value_array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| MahoutError::Io("List values must be Float64".to_string()))?;

                let current_size = float_array.len();

                if let Some(expected_size) = sample_size {
                    if current_size != expected_size {
                        return Err(MahoutError::InvalidInput(format!(
                            "Inconsistent sample sizes: expected {}, got {}",
                            expected_size, current_size
                        )));
                    }
                } else {
                    sample_size = Some(current_size);
                    all_data.reserve(current_size * total_rows);
                }

                if float_array.null_count() == 0 {
                    all_data.extend_from_slice(float_array.values());
                } else {
                    all_data.extend(float_array.iter().map(|opt| opt.unwrap_or(0.0)));
                }

                num_samples += 1;
            }
        } else {
            return Err(MahoutError::Io(format!(
                "Expected List<Float64> column, got {:?}",
                column.data_type()
            )));
        }
    }

    let sample_size = sample_size.ok_or_else(|| {
        MahoutError::Io("Parquet file contains no data".to_string())
    })?;

    Ok((all_data, num_samples, sample_size))
}

/// Reads batch data from an Arrow IPC file.
///
/// Supports `FixedSizeList<Float64>` and `List<Float64>` column formats.
/// Returns flattened data suitable for batch encoding.
///
/// # Returns
/// Tuple of `(flattened_data, num_samples, sample_size)`
///
/// # TODO
/// Add OOM protection for very large files
pub fn read_arrow_ipc_batch<P: AsRef<Path>>(path: P) -> Result<(Vec<f64>, usize, usize)> {
    let file = File::open(path.as_ref()).map_err(|e| {
        MahoutError::Io(format!("Failed to open Arrow IPC file: {}", e))
    })?;

    let reader = ArrowFileReader::try_new(file, None).map_err(|e| {
        MahoutError::Io(format!("Failed to create Arrow IPC reader: {}", e))
    })?;

    let mut all_data = Vec::new();
    let mut num_samples = 0;
    let mut sample_size: Option<usize> = None;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            MahoutError::Io(format!("Failed to read Arrow batch: {}", e))
        })?;

        if batch.num_columns() == 0 {
            return Err(MahoutError::Io("Arrow file has no columns".to_string()));
        }

        let column = batch.column(0);

        match column.data_type() {
            DataType::FixedSizeList(_, size) => {
                let list_array = column
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| MahoutError::Io("Failed to downcast to FixedSizeListArray".to_string()))?;

                let current_size = *size as usize;

                if let Some(expected) = sample_size {
                    if current_size != expected {
                        return Err(MahoutError::InvalidInput(format!(
                            "Inconsistent sample sizes: expected {}, got {}",
                            expected, current_size
                        )));
                    }
                } else {
                    sample_size = Some(current_size);
                    all_data.reserve(current_size * batch.num_rows());
                }

                let values = list_array.values();
                let float_array = values
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| MahoutError::Io("Values must be Float64".to_string()))?;

                if float_array.null_count() == 0 {
                    all_data.extend_from_slice(float_array.values());
                } else {
                    all_data.extend(float_array.iter().map(|opt| opt.unwrap_or(0.0)));
                }

                num_samples += list_array.len();
            }

            DataType::List(_) => {
                let list_array = column
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| MahoutError::Io("Failed to downcast to ListArray".to_string()))?;

                for i in 0..list_array.len() {
                    let value_array = list_array.value(i);
                    let float_array = value_array
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| MahoutError::Io("List values must be Float64".to_string()))?;

                    let current_size = float_array.len();

                    if let Some(expected) = sample_size {
                        if current_size != expected {
                            return Err(MahoutError::InvalidInput(format!(
                                "Inconsistent sample sizes: expected {}, got {}",
                                expected, current_size
                            )));
                        }
                    } else {
                        sample_size = Some(current_size);
                        all_data.reserve(current_size * list_array.len());
                    }

                    if float_array.null_count() == 0 {
                        all_data.extend_from_slice(float_array.values());
                    } else {
                        all_data.extend(float_array.iter().map(|opt| opt.unwrap_or(0.0)));
                    }

                    num_samples += 1;
                }
            }

            _ => {
                return Err(MahoutError::Io(format!(
                    "Expected FixedSizeList<Float64> or List<Float64>, got {:?}",
                    column.data_type()
                )));
            }
        }
    }

    let sample_size = sample_size.ok_or_else(|| {
        MahoutError::Io("Arrow file contains no data".to_string())
    })?;

    Ok((all_data, num_samples, sample_size))
}

/// Streaming Parquet reader for List<Float64> and FixedSizeList<Float64> columns
///
/// Reads Parquet files in chunks without loading entire file into memory.
/// Supports efficient streaming for large files via Producer-Consumer pattern.
pub struct ParquetBlockReader {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    sample_size: Option<usize>,
    leftover_data: Vec<f64>,
    leftover_cursor: usize,
    pub total_rows: usize,
}

impl ParquetBlockReader {
    /// Create a new streaming Parquet reader
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Optional batch size (defaults to 2048)
    pub fn new<P: AsRef<Path>>(path: P, batch_size: Option<usize>) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            MahoutError::Io(format!("Failed to open Parquet file: {}", e))
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            MahoutError::Io(format!("Failed to create Parquet reader: {}", e))
        })?;

        let schema = builder.schema();
        if schema.fields().len() != 1 {
            return Err(MahoutError::InvalidInput(format!(
                "Expected exactly one column, got {}",
                schema.fields().len()
            )));
        }

        let field = &schema.fields()[0];
        match field.data_type() {
            DataType::List(child_field) => {
                if !matches!(child_field.data_type(), DataType::Float64) {
                    return Err(MahoutError::InvalidInput(format!(
                        "Expected List<Float64> column, got List<{:?}>",
                        child_field.data_type()
                    )));
                }
            }
            DataType::FixedSizeList(child_field, _) => {
                if !matches!(child_field.data_type(), DataType::Float64) {
                    return Err(MahoutError::InvalidInput(format!(
                        "Expected FixedSizeList<Float64> column, got FixedSizeList<{:?}>",
                        child_field.data_type()
                    )));
                }
            }
            _ => {
                return Err(MahoutError::InvalidInput(format!(
                    "Expected List<Float64> or FixedSizeList<Float64> column, got {:?}",
                    field.data_type()
                )));
            }
        }

        let total_rows = builder.metadata().file_metadata().num_rows() as usize;

        let batch_size = batch_size.unwrap_or(2048);
        let reader = builder
            .with_batch_size(batch_size)
            .build()
            .map_err(|e| {
                MahoutError::Io(format!("Failed to build Parquet reader: {}", e))
            })?;

        Ok(Self {
            reader,
            sample_size: None,
            leftover_data: Vec::new(),
            leftover_cursor: 0,
            total_rows,
        })
    }

    /// Get the sample size (number of elements per sample)
    pub fn get_sample_size(&self) -> Option<usize> {
        self.sample_size
    }

    /// Read a chunk of data into the provided buffer
    ///
    /// Handles leftover data from previous reads and ensures sample boundaries are respected.
    /// Returns the number of elements written to the buffer.
    pub fn read_chunk(&mut self, buffer: &mut [f64]) -> Result<usize> {
        let mut written = 0;
        let buf_cap = buffer.len();
        let calc_limit = |ss: usize| -> usize {
            if ss == 0 {
                buf_cap
            } else {
                (buf_cap / ss) * ss
            }
        };
        let mut limit = self.sample_size.map_or(buf_cap, calc_limit);

        if self.sample_size.is_some() {
            while self.leftover_cursor < self.leftover_data.len() && written < limit {
                let available = self.leftover_data.len() - self.leftover_cursor;
                let space_left = limit - written;
                let to_copy = std::cmp::min(available, space_left);

                if to_copy > 0 {
                    buffer[written..written+to_copy].copy_from_slice(
                        &self.leftover_data[self.leftover_cursor..self.leftover_cursor+to_copy]
                    );
                    written += to_copy;
                    self.leftover_cursor += to_copy;

                    if self.leftover_cursor == self.leftover_data.len() {
                        self.leftover_data.clear();
                        self.leftover_cursor = 0;
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        while written < limit {
            match self.reader.next() {
                Some(Ok(batch)) => {
                    if batch.num_columns() == 0 {
                        continue;
                    }
                    let column = batch.column(0);

                    let mut push_values = |values: &[f64]| -> Result<bool> {
                        // returns true if buffer filled and we should break outer loop
                        let available = values.len();
                        let space_left = limit - written;

                        if available <= space_left {
                            buffer[written..written+available].copy_from_slice(values);
                            written += available;
                            Ok(false)
                        } else {
                            if space_left > 0 {
                                buffer[written..written+space_left].copy_from_slice(&values[0..space_left]);
                                written += space_left;
                            }
                            self.leftover_data.clear();
                            self.leftover_data.extend_from_slice(&values[space_left..]);
                            self.leftover_cursor = 0;
                            Ok(true)
                        }
                    };

                    let current_sample_size = match column.data_type() {
                        DataType::List(_) => {
                            let list_array = column
                                .as_any()
                                .downcast_ref::<ListArray>()
                                .ok_or_else(|| MahoutError::Io("Failed to downcast to ListArray".to_string()))?;

                            if list_array.len() == 0 {
                                continue;
                            }

                            let mut detected_size = None;
                            for i in 0..list_array.len() {
                                let value_array = list_array.value(i);
                                let float_array = value_array
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                    .ok_or_else(|| MahoutError::Io("List values must be Float64".to_string()))?;

                                if float_array.null_count() != 0 {
                                    return Err(MahoutError::Io("Null value encountered in Float64Array during quantum encoding. Please check data quality at the source.".to_string()));
                                }

                                let len = float_array.len();
                                if detected_size.is_none() {
                                    detected_size = Some(len);
                                } else if Some(len) != detected_size {
                                    return Err(MahoutError::InvalidInput(format!(
                                        "Inconsistent sample sizes: expected {}, got {}",
                                        detected_size.unwrap(), len
                                    )));
                                }

                                let should_break = push_values(float_array.values())?;
                                if should_break {
                                    break;
                                }
                            }

                            detected_size.expect("list_array.len() > 0 ensures at least one element")
                        }
                        DataType::FixedSizeList(_, size) => {
                            let list_array = column
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .ok_or_else(|| MahoutError::Io("Failed to downcast to FixedSizeListArray".to_string()))?;

                            if list_array.len() == 0 {
                                continue;
                            }

                            let current_sample_size = *size as usize;

                            let values = list_array.values();
                            let float_array = values
                                .as_any()
                                .downcast_ref::<Float64Array>()
                                .ok_or_else(|| MahoutError::Io("FixedSizeList values must be Float64".to_string()))?;

                            if float_array.null_count() != 0 {
                                return Err(MahoutError::Io("Null value encountered in Float64Array during quantum encoding. Please check data quality at the source.".to_string()));
                            }

                            let _ = push_values(float_array.values())?;
                            current_sample_size
                        }
                        _ => {
                            return Err(MahoutError::Io(format!(
                                "Expected List<Float64> or FixedSizeList<Float64>, got {:?}",
                                column.data_type()
                            )));
                        }
                    };

                    if self.sample_size.is_none() {
                        self.sample_size = Some(current_sample_size);
                        limit = calc_limit(current_sample_size);
                    } else {
                        if let Some(expected_size) = self.sample_size {
                            if current_sample_size != expected_size {
                                return Err(MahoutError::InvalidInput(format!(
                                    "Inconsistent sample sizes: expected {}, got {}",
                                    expected_size, current_sample_size
                                )));
                            }
                        }
                    }
                },
                Some(Err(e)) => return Err(MahoutError::Io(format!("Parquet read error: {}", e))),
                None => break,
            }
        }

        Ok(written)
    }
}
