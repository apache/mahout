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

//! Parquet format reader implementation.

use std::fs::File;
use std::path::Path;

use arrow::array::{Array, FixedSizeListArray, Float64Array, ListArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::{MahoutError, Result};
use crate::reader::{DataReader, StreamingDataReader};

/// Reader for Parquet files containing List<Float64> or FixedSizeList<Float64> columns.
pub struct ParquetReader {
    reader: Option<parquet::arrow::arrow_reader::ParquetRecordBatchReader>,
    sample_size: Option<usize>,
    total_rows: usize,
}

impl ParquetReader {
    /// Create a new Parquet reader.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Optional batch size for reading (defaults to entire file)
    pub fn new<P: AsRef<Path>>(path: P, batch_size: Option<usize>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| MahoutError::Io(format!("Failed to open Parquet file: {}", e)))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| MahoutError::Io(format!("Failed to create Parquet reader: {}", e)))?;

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

        let reader = if let Some(batch_size) = batch_size {
            builder.with_batch_size(batch_size).build()
        } else {
            builder.build()
        }
        .map_err(|e| MahoutError::Io(format!("Failed to build Parquet reader: {}", e)))?;

        Ok(Self {
            reader: Some(reader),
            sample_size: None,
            total_rows,
        })
    }
}

impl DataReader for ParquetReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        let reader = self
            .reader
            .take()
            .ok_or_else(|| MahoutError::InvalidInput("Reader already consumed".to_string()))?;

        let mut all_data = Vec::new();
        let mut num_samples = 0;
        let mut sample_size = None;

        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| MahoutError::Io(format!("Failed to read Parquet batch: {}", e)))?;

            if batch.num_columns() == 0 {
                return Err(MahoutError::Io("Parquet file has no columns".to_string()));
            }

            let column = batch.column(0);

            match column.data_type() {
                DataType::List(_) => {
                    let list_array =
                        column.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                            MahoutError::Io("Failed to downcast to ListArray".to_string())
                        })?;

                    for i in 0..list_array.len() {
                        let value_array = list_array.value(i);
                        let float_array = value_array
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| {
                                MahoutError::Io("List values must be Float64".to_string())
                            })?;

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
                            all_data.reserve(current_size * self.total_rows);
                        }

                        if float_array.null_count() == 0 {
                            all_data.extend_from_slice(float_array.values());
                        } else {
                            all_data.extend(float_array.iter().map(|opt| opt.unwrap_or(0.0)));
                        }

                        num_samples += 1;
                    }
                }
                DataType::FixedSizeList(_, size) => {
                    let list_array = column
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .ok_or_else(|| {
                            MahoutError::Io("Failed to downcast to FixedSizeListArray".to_string())
                        })?;

                    let current_size = *size as usize;

                    if sample_size.is_none() {
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
                _ => {
                    return Err(MahoutError::Io(format!(
                        "Expected List<Float64> or FixedSizeList<Float64>, got {:?}",
                        column.data_type()
                    )));
                }
            }
        }

        let sample_size = sample_size
            .ok_or_else(|| MahoutError::Io("Parquet file contains no data".to_string()))?;

        self.sample_size = Some(sample_size);

        Ok((all_data, num_samples, sample_size))
    }

    fn get_sample_size(&self) -> Option<usize> {
        self.sample_size
    }

    fn get_num_samples(&self) -> Option<usize> {
        Some(self.total_rows)
    }
}

/// Streaming Parquet reader for List<Float64> and FixedSizeList<Float64> columns.
///
/// Reads Parquet files in chunks without loading entire file into memory.
/// Supports efficient streaming for large files via Producer-Consumer pattern.
pub struct ParquetStreamingReader {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    sample_size: Option<usize>,
    leftover_data: Vec<f64>,
    leftover_cursor: usize,
    pub total_rows: usize,
}

impl ParquetStreamingReader {
    /// Create a new streaming Parquet reader.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Optional batch size (defaults to 2048)
    pub fn new<P: AsRef<Path>>(path: P, batch_size: Option<usize>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| MahoutError::Io(format!("Failed to open Parquet file: {}", e)))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| MahoutError::Io(format!("Failed to create Parquet reader: {}", e)))?;

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
            .map_err(|e| MahoutError::Io(format!("Failed to build Parquet reader: {}", e)))?;

        Ok(Self {
            reader,
            sample_size: None,
            leftover_data: Vec::new(),
            leftover_cursor: 0,
            total_rows,
        })
    }

    /// Get the sample size (number of elements per sample).
    pub fn get_sample_size(&self) -> Option<usize> {
        self.sample_size
    }
}

impl DataReader for ParquetStreamingReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        let mut all_data = Vec::new();
        let mut num_samples = 0;

        loop {
            let mut buffer = vec![0.0; 1024 * 1024]; // 1M elements buffer
            let written = self.read_chunk(&mut buffer)?;
            if written == 0 {
                break;
            }
            all_data.extend_from_slice(&buffer[..written]);
            num_samples += written / self.sample_size.unwrap_or(1);
        }

        let sample_size = self
            .sample_size
            .ok_or_else(|| MahoutError::Io("No data read from Parquet file".to_string()))?;

        Ok((all_data, num_samples, sample_size))
    }

    fn get_sample_size(&self) -> Option<usize> {
        self.sample_size
    }

    fn get_num_samples(&self) -> Option<usize> {
        Some(self.total_rows)
    }
}

impl StreamingDataReader for ParquetStreamingReader {
    fn read_chunk(&mut self, buffer: &mut [f64]) -> Result<usize> {
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
                    buffer[written..written + to_copy].copy_from_slice(
                        &self.leftover_data[self.leftover_cursor..self.leftover_cursor + to_copy],
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

                    let (current_sample_size, batch_values) = match column.data_type() {
                        DataType::List(_) => {
                            let list_array =
                                column.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                                    MahoutError::Io("Failed to downcast to ListArray".to_string())
                                })?;

                            if list_array.len() == 0 {
                                continue;
                            }

                            let mut batch_values = Vec::new();
                            let mut current_sample_size = None;
                            for i in 0..list_array.len() {
                                let value_array = list_array.value(i);
                                let float_array = value_array
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                    .ok_or_else(|| {
                                        MahoutError::Io("List values must be Float64".to_string())
                                    })?;

                                if i == 0 {
                                    current_sample_size = Some(float_array.len());
                                }

                                if float_array.null_count() == 0 {
                                    batch_values.extend_from_slice(float_array.values());
                                } else {
                                    return Err(MahoutError::Io("Null value encountered in Float64Array during quantum encoding. Please check data quality at the source.".to_string()));
                                }
                            }

                            (
                                current_sample_size
                                    .expect("list_array.len() > 0 ensures at least one element"),
                                batch_values,
                            )
                        }
                        DataType::FixedSizeList(_, size) => {
                            let list_array = column
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .ok_or_else(|| {
                                MahoutError::Io(
                                    "Failed to downcast to FixedSizeListArray".to_string(),
                                )
                            })?;

                            if list_array.len() == 0 {
                                continue;
                            }

                            let current_sample_size = *size as usize;

                            let values = list_array.values();
                            let float_array = values
                                .as_any()
                                .downcast_ref::<Float64Array>()
                                .ok_or_else(|| {
                                    MahoutError::Io(
                                        "FixedSizeList values must be Float64".to_string(),
                                    )
                                })?;

                            let mut batch_values = Vec::new();
                            if float_array.null_count() == 0 {
                                batch_values.extend_from_slice(float_array.values());
                            } else {
                                return Err(MahoutError::Io("Null value encountered in Float64Array during quantum encoding. Please check data quality at the source.".to_string()));
                            }

                            (current_sample_size, batch_values)
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
                    } else if let Some(expected_size) = self.sample_size
                        && current_sample_size != expected_size
                    {
                        return Err(MahoutError::InvalidInput(format!(
                            "Inconsistent sample sizes: expected {}, got {}",
                            expected_size, current_sample_size
                        )));
                    }

                    let available = batch_values.len();
                    let space_left = limit - written;

                    if available <= space_left {
                        buffer[written..written + available].copy_from_slice(&batch_values);
                        written += available;
                    } else {
                        if space_left > 0 {
                            buffer[written..written + space_left]
                                .copy_from_slice(&batch_values[0..space_left]);
                            written += space_left;
                        }
                        self.leftover_data.clear();
                        self.leftover_data
                            .extend_from_slice(&batch_values[space_left..]);
                        self.leftover_cursor = 0;
                        break;
                    }
                }
                Some(Err(e)) => return Err(MahoutError::Io(format!("Parquet read error: {}", e))),
                None => break,
            }
        }

        Ok(written)
    }

    fn total_rows(&self) -> usize {
        self.total_rows
    }
}
