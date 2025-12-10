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

//! I/O module for reading and writing quantum data
//!
//! This module provides efficient columnar data exchange with the data science ecosystem,

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, ListArray, RecordBatch, AsArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::error::{MahoutError, Result};

/// Convert Arrow Float64Array to Vec<f64>
///
/// Uses Arrow's internal buffer if no nulls, otherwise copies
pub fn arrow_to_vec(array: &Float64Array) -> Vec<f64> {
    if array.null_count() == 0 {
        array.values().to_vec()
    } else {
        array.iter().map(|opt| opt.unwrap_or(0.0)).collect()
    }
}

/// Convert chunked Arrow Float64Array to Vec<f64>
///
/// Efficiently flattens multiple Arrow arrays into a single Vec
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

/// Reads quantum data from a Parquet file.
///
/// Expects a single column named "data" containing Float64 values.
/// This function performs one copy from Arrow to Vec.
/// use `read_parquet_to_arrow` instead.
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
    let chunks = read_parquet_to_arrow(path)?;
    Ok(arrow_to_vec_chunked(&chunks))
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
    let array = Float64Array::from_iter_values(data.iter().copied());
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

/// Reads quantum data from a Parquet file as Arrow arrays.
///
/// Returns Arrow arrays from Parquet batches.
/// Each element in the returned Vec corresponds to one Parquet batch.
///
/// Constructs the Arrow array from Parquet batches
///
/// # Arguments
/// * `path` - Path to the Parquet file
///
/// # Returns
/// Vector of Float64Arrays, one per Parquet batch
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

        // Clone the Float64Array (reference-counted, no data copy)
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
/// Writes from Arrow format to Parquet.
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

/// Read batch data from Parquet file with list column format
///
/// Efficiently reads Parquet files where each row contains a list of values.
/// Returns a flattened Vec with all samples concatenated, suitable for batch encoding.
///
/// # Arguments
/// * `path` - Path to Parquet file
///
/// # Returns
/// Tuple of (flattened_data, num_samples, sample_size)
///
/// # Example
/// File format: column "feature_vector" with type List<Float64>
/// Each row = one sample = one list of floats
pub fn read_parquet_batch<P: AsRef<Path>>(path: P) -> Result<(Vec<f64>, usize, usize)> {
    let file = File::open(path.as_ref()).map_err(|e| {
        MahoutError::Io(format!("Failed to open Parquet file: {}", e))
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        MahoutError::Io(format!("Failed to create Parquet reader: {}", e))
    })?;

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

        // Handle List<Float64> column type
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

                // Verify all samples have the same size
                if let Some(expected_size) = sample_size {
                    if current_size != expected_size {
                        return Err(MahoutError::InvalidInput(format!(
                            "Inconsistent sample sizes: expected {}, got {}",
                            expected_size, current_size
                        )));
                    }
                } else {
                    sample_size = Some(current_size);
                    all_data.reserve(current_size * 100); // Reserve space
                }

                // Efficiently copy the values
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

// === Streaming IO Support ===

/// Streaming Parquet reader for pipeline processing
///
/// Reads chunks into staging buffers, avoiding full-file memory allocation.
pub struct ParquetBlockReader {
    reader: ParquetRecordBatchReader,
    sample_size: Option<usize>,
    leftover_data: Vec<f64>,
    leftover_cursor: usize,
    pub total_samples: usize,
}

impl ParquetBlockReader {
    /// Initialize reader and validate schema
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            MahoutError::Io(format!("Failed to open Parquet file: {}", e))
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            MahoutError::Io(format!("Failed to create Parquet reader: {}", e))
        })?;

        let total_rows = builder.metadata().file_metadata().num_rows() as usize;
        let reader = builder.build().map_err(|e| {
            MahoutError::Io(format!("Failed to build Parquet reader: {}", e))
        })?;

        Ok(Self {
            reader,
            sample_size: None,
            leftover_data: Vec::new(),
            leftover_cursor: 0,
            total_samples: total_rows,
        })
    }

    /// Read next chunk into the provided buffer.
    /// Ensures only complete samples are written if sample_size is known.
    /// Returns the number of f64 elements written.
    pub fn read_chunk(&mut self, buffer: &mut [f64]) -> Result<usize> {
        let mut written = 0;
        let buf_cap = buffer.len();

        // Calculate capacity aligned to sample boundaries
        let calc_limit = |ss: usize| -> usize {
            if ss == 0 { buf_cap } else { (buf_cap / ss) * ss }
        };

        let mut limit = self.sample_size.map_or(buf_cap, calc_limit);

        if limit == 0 && self.sample_size.is_some() {
            return Err(MahoutError::MemoryAllocation(format!(
                "Staging buffer too small ({} elements) for sample size ({} elements)",
                buf_cap, self.sample_size.unwrap()
            )));
        }

        // Drain leftovers first
        if self.leftover_cursor < self.leftover_data.len() {
            let available = self.leftover_data.len() - self.leftover_cursor;
            let space_left = limit - written;

            // Ensure we only copy complete samples if sample_size is known
            let to_copy = if let Some(ss) = self.sample_size {
                if ss == 0 {
                    std::cmp::min(available, space_left)
                } else {
                    // Align to sample boundaries
                    let max_by_space = (space_left / ss) * ss;
                    let max_by_available = (available / ss) * ss;
                    std::cmp::min(max_by_available, max_by_space)
                }
            } else {
                std::cmp::min(available, space_left)
            };

            if to_copy > 0 {
                buffer[written..written+to_copy].copy_from_slice(&self.leftover_data[self.leftover_cursor..self.leftover_cursor+to_copy]);

                written += to_copy;
                self.leftover_cursor += to_copy;

                if self.leftover_cursor == self.leftover_data.len() {
                    self.leftover_data.clear();
                    self.leftover_cursor = 0;
                }
            }

            if written == limit {
                return Ok(written);
            }
        }

        // Read new batches until buffer is full
        while written < limit {
            match self.reader.next() {
                Some(Ok(batch)) => {
                    if batch.num_columns() == 0 { continue; }
                    let column = batch.column(0);

                    let (values, current_sample_size) = extract_values_bulk(column)?;

                    // On first read, establish sample size
                    if self.sample_size.is_none() {
                        self.sample_size = Some(current_sample_size);
                        let new_limit = calc_limit(current_sample_size);
                        if written > new_limit {
                            self.leftover_data.clear();
                            self.leftover_data.extend_from_slice(&buffer[new_limit..written]);
                            written = new_limit;
                            self.leftover_cursor = 0;
                            return Ok(written);
                        }
                        limit = new_limit;
                    } else if current_sample_size != self.sample_size.unwrap() {
                        return Err(MahoutError::InvalidInput(format!(
                            "Inconsistent sample size: expected {}, got {}",
                            self.sample_size.unwrap(), current_sample_size
                        )));
                    }

                    let available = values.len();
                    let space_left = if written >= limit { 0 } else { limit - written };

                    if available <= space_left {
                        buffer[written..written+available].copy_from_slice(values);
                        written += available;
                    } else {
                        if space_left > 0 {
                            buffer[written..written+space_left].copy_from_slice(&values[0..space_left]);
                            written += space_left;
                        }

                        self.leftover_data.clear();
                        self.leftover_data.extend_from_slice(&values[space_left..]);
                        self.leftover_cursor = 0;
                        break;
                    }
                },
                Some(Err(e)) => return Err(MahoutError::Io(format!("Parquet read error: {}", e))),
                None => break,
            }
        }

        Ok(written)
    }

    pub fn get_sample_size(&self) -> Option<usize> {
        self.sample_size
    }
}

/// Extract Arrow List<Float64> values directly from underlying buffers.
/// Returns (values slice, sample size per row).
fn extract_values_bulk(column: &ArrayRef) -> Result<(&[f64], usize)> {
    match column.data_type() {
        DataType::List(_) => {
            let list_array = column.as_list::<i32>();
            let values_array = list_array.values();
            let float_values = values_array.as_primitive::<arrow::datatypes::Float64Type>();
            let raw_slice = float_values.values();

            if list_array.is_empty() {
                return Ok((&[], 0));
            }

            let offsets = list_array.value_offsets();
            let sample_size = if offsets.len() > 1 {
                (offsets[1] - offsets[0]) as usize
            } else {
                0
            };

            let start_offset = offsets[0] as usize;
            let end_offset = offsets[offsets.len() - 1] as usize;

            if end_offset > raw_slice.len() {
                 return Err(MahoutError::Io("Corrupt Arrow Array: Offsets exceed value buffer".into()));
            }

            Ok((&raw_slice[start_offset..end_offset], sample_size))
        },
        _ => Err(MahoutError::Io(format!("Expected List<Float64>, got {:?}", column.data_type()))),
    }
}

/// Legacy: Extract to Vec (kept for compatibility).
#[allow(dead_code)]
fn extract_batch_data(column: &ArrayRef, expected_size: Option<usize>) -> Result<(Vec<f64>, usize)> {
    let mut batch_data = Vec::new();
    let mut sample_size = expected_size.unwrap_or(0);

    let list_array = column.as_any().downcast_ref::<ListArray>()
        .ok_or_else(|| MahoutError::Io(format!("Expected List<Float64>, got {:?}", column.data_type())))?;

    for i in 0..list_array.len() {
        let value_array = list_array.value(i);
        let float_array = value_array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| MahoutError::Io("List values must be Float64".to_string()))?;

        let current_len = float_array.len();

        if let Some(expected) = expected_size {
            if current_len != expected {
                return Err(MahoutError::InvalidInput(format!(
                    "Inconsistent sample sizes: expected {}, got {}", expected, current_len
                )));
            }
        } else if sample_size == 0 {
            sample_size = current_len;
        }

        if float_array.null_count() == 0 {
            batch_data.extend_from_slice(float_array.values());
        } else {
            batch_data.extend(float_array.iter().map(|opt| opt.unwrap_or(0.0)));
        }
    }

    Ok((batch_data, sample_size))
}
