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
use crate::reader::{DataReader, NullHandling, StreamingDataReader, handle_float64_nulls};

/// Reader for Parquet files containing List<Float64> or FixedSizeList<Float64> columns.
pub struct ParquetReader {
    reader: Option<parquet::arrow::arrow_reader::ParquetRecordBatchReader>,
    sample_size: Option<usize>,
    total_rows: usize,
    null_handling: NullHandling,
}

impl ParquetReader {
    /// Create a new Parquet reader.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Optional batch size for reading (defaults to entire file)
    /// * `null_handling` - Policy for null values (defaults to `FillZero`)
    pub fn new<P: AsRef<Path>>(
        path: P,
        batch_size: Option<usize>,
        null_handling: NullHandling,
    ) -> Result<Self> {
        let path = path.as_ref();

        // Verify file exists
        match path.try_exists() {
            Ok(false) => {
                return Err(MahoutError::Io(format!(
                    "Parquet file not found: {}",
                    path.display()
                )));
            }
            Err(e) => {
                return Err(MahoutError::IoWithSource {
                    message: format!(
                        "Failed to check if Parquet file exists at {}: {}",
                        path.display(),
                        e
                    ),
                    source: e,
                });
            }
            Ok(true) => {}
        }

        let file = File::open(path).map_err(|e| MahoutError::IoWithSource {
            message: format!("Failed to open Parquet file: {}", e),
            source: e,
        })?;

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
            null_handling,
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

                        handle_float64_nulls(&mut all_data, float_array, self.null_handling)?;

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

                    handle_float64_nulls(&mut all_data, float_array, self.null_handling)?;

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
    null_handling: NullHandling,
}

impl ParquetStreamingReader {
    /// Create a new streaming Parquet reader.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Optional batch size (defaults to 2048)
    /// * `null_handling` - Policy for null values (defaults to `FillZero`)
    pub fn new<P: AsRef<Path>>(
        path: P,
        batch_size: Option<usize>,
        null_handling: NullHandling,
    ) -> Result<Self> {
        let path = path.as_ref();

        // Verify file exists
        match path.try_exists() {
            Ok(false) => {
                return Err(MahoutError::Io(format!(
                    "Parquet file not found: {}",
                    path.display()
                )));
            }
            Err(e) => {
                return Err(MahoutError::IoWithSource {
                    message: format!(
                        "Failed to check if Parquet file exists at {}: {}",
                        path.display(),
                        e
                    ),
                    source: e,
                });
            }
            Ok(true) => {}
        }

        let file = File::open(path).map_err(|e| MahoutError::IoWithSource {
            message: format!("Failed to open Parquet file: {}", e),
            source: e,
        })?;

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
            DataType::Float64 => {
                // Scalar Float64 for basis encoding (one index per sample)
            }
            _ => {
                return Err(MahoutError::InvalidInput(format!(
                    "Expected Float64, List<Float64>, or FixedSizeList<Float64> column, got {:?}",
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
            null_handling,
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

                                handle_float64_nulls(
                                    &mut batch_values,
                                    float_array,
                                    self.null_handling,
                                )?;
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
                            handle_float64_nulls(
                                &mut batch_values,
                                float_array,
                                self.null_handling,
                            )?;

                            (current_sample_size, batch_values)
                        }
                        DataType::Float64 => {
                            // Scalar Float64 for basis encoding (one index per sample)
                            let float_array = column
                                .as_any()
                                .downcast_ref::<Float64Array>()
                                .ok_or_else(|| {
                                    MahoutError::Io(
                                        "Failed to downcast to Float64Array".to_string(),
                                    )
                                })?;

                            if float_array.is_empty() {
                                continue;
                            }

                            let current_sample_size = 1;

                            let mut batch_values = Vec::new();
                            handle_float64_nulls(
                                &mut batch_values,
                                float_array,
                                self.null_handling,
                            )?;

                            (current_sample_size, batch_values)
                        }
                        _ => {
                            return Err(MahoutError::Io(format!(
                                "Expected Float64, List<Float64>, or FixedSizeList<Float64>, got {:?}",
                                column.data_type()
                            )));
                        }
                    };

                    if self.sample_size.is_none() {
                        self.sample_size = Some(current_sample_size);
                        limit = calc_limit(current_sample_size);
                    } else if current_sample_size != self.sample_size.unwrap() {
                        return Err(MahoutError::InvalidInput(format!(
                            "Inconsistent sample sizes: expected {}, got {}",
                            self.sample_size.unwrap(), current_sample_size
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
#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, FixedSizeListBuilder, Float64Builder, Int32Array, ListBuilder, RecordBatch,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::fs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempTestFile {
        path: std::path::PathBuf,
    }

    impl TempTestFile {
        fn new() -> Self {
            let count = TEST_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "mahout_test_parquet_{}_{}.parquet",
                std::process::id(),
                count
            ));
            Self { path }
        }

        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TempTestFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn write_test_parquet(schema: Arc<Schema>, arrays: Vec<ArrayRef>) -> TempTestFile {
        let file = TempTestFile::new();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        let os_file = fs::File::create(file.path()).unwrap();
        let mut writer = ArrowWriter::try_new(os_file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        file
    }

    #[test]
    fn test_parquet_reader_missing_file() {
        let path = std::env::temp_dir().join(format!("missing_{}.parquet", std::process::id()));
        let result = ParquetReader::new(&path, None, NullHandling::FillZero);
        assert!(matches!(result, Err(MahoutError::Io(_))));
    }

    #[test]
    fn test_parquet_reader_bad_schema() {
        let schema = Arc::new(Schema::new(vec![Field::new("bad", DataType::Int32, false)]));
        let array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let file = write_test_parquet(schema, vec![array]);

        let result = ParquetReader::new(file.path(), None, NullHandling::FillZero);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Expected List<Float64> or FixedSizeList<Float64> column"));
    }

    #[test]
    fn test_parquet_reader_too_many_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col1", DataType::Int32, false),
            Field::new("col2", DataType::Int32, false),
        ]));
        let arr1 = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let arr2 = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let file = write_test_parquet(schema, vec![arr1, arr2]);

        let result = ParquetReader::new(file.path(), None, NullHandling::FillZero);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Expected exactly one column"));
    }

    #[test]
    fn test_parquet_reader_bad_list_type() {
        let item_field = Arc::new(Field::new("item", DataType::Int32, true));
        let list_field = Field::new("list_col", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = arrow::array::ListBuilder::new(arrow::array::Int32Builder::new());
        builder.values().append_value(1);
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        let file = write_test_parquet(schema, vec![array]);

        let result = ParquetReader::new(file.path(), None, NullHandling::FillZero);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Expected List<Float64> column"));
    }

    #[test]
    fn test_parquet_reader_list_f64() {
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = ListBuilder::new(Float64Builder::new());
        builder.values().append_slice(&[1.0, 2.0]);
        builder.append(true);
        builder.values().append_slice(&[3.0, 4.0]);
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        let file = write_test_parquet(schema, vec![array]);

        let mut reader = ParquetReader::new(file.path(), None, NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(num_samples, 2);
        assert_eq!(sample_size, 2);
        assert_eq!(reader.get_sample_size(), Some(2));
        assert_eq!(reader.get_num_samples(), Some(2));
    }

    #[test]
    fn test_parquet_reader_fixed_size_list_f64() {
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::FixedSizeList(item_field.clone(), 2), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), 2);
        builder.values().append_slice(&[5.0, 6.0]);
        builder.append(true);
        builder.values().append_slice(&[7.0, 8.0]);
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        let file = write_test_parquet(schema, vec![array]);

        let mut reader = ParquetReader::new(file.path(), None, NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(num_samples, 2);
        assert_eq!(sample_size, 2);
    }

    #[test]
    fn test_parquet_reader_inconsistent_sample_sizes() {
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = ListBuilder::new(Float64Builder::new());
        builder.values().append_slice(&[1.0, 2.0]);
        builder.append(true);
        builder.values().append_slice(&[3.0, 4.0, 5.0]); // Inconsistent!
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        let file = write_test_parquet(schema, vec![array]);

        let mut reader = ParquetReader::new(file.path(), None, NullHandling::FillZero).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Inconsistent sample sizes: expected 2, got 3"));
    }

    #[test]
    fn test_parquet_streaming_reader_scalar_f64() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Float64,
            false,
        )]));
        let mut builder = Float64Builder::new();
        builder.append_slice(&[10.0, 20.0, 30.0]);
        let array = Arc::new(builder.finish()) as ArrayRef;
        let file = write_test_parquet(schema, vec![array]);

        let mut streaming_reader =
            ParquetStreamingReader::new(file.path(), None, NullHandling::FillZero).unwrap();
        assert_eq!(streaming_reader.total_rows(), 3);
        let mut buffer = vec![0.0; 2];
        let written1 = streaming_reader.read_chunk(&mut buffer).unwrap();
        assert_eq!(written1, 2);
        assert_eq!(buffer, vec![10.0, 20.0]);

        let mut buffer2 = vec![0.0; 2];
        let written2 = streaming_reader.read_chunk(&mut buffer2).unwrap();
        assert_eq!(written2, 1);
        assert_eq!(buffer2[..1], vec![30.0]);

        let written3 = streaming_reader.read_chunk(&mut buffer2).unwrap();
        assert_eq!(written3, 0);
    }

    #[test]
    fn test_parquet_streaming_reader_list_f64_chunked() {
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = ListBuilder::new(Float64Builder::new());
        builder.values().append_slice(&[1.0, 2.0]);
        builder.append(true);
        builder.values().append_slice(&[3.0, 4.0]);
        builder.append(true);
        builder.values().append_slice(&[5.0, 6.0]);
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        let file = write_test_parquet(schema, vec![array]);

        let mut reader =
            ParquetStreamingReader::new(file.path(), Some(1), NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(num_samples, 3);
        assert_eq!(sample_size, 2);
    }

    #[test]
    fn test_parquet_reader_empty_data() {
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = ListBuilder::new(Float64Builder::new());
        let array = Arc::new(builder.finish()) as ArrayRef;
        let file = write_test_parquet(schema, vec![array]);

        let mut reader = ParquetReader::new(file.path(), None, NullHandling::FillZero).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("no data") || err_msg.contains("no columns"));
    }

    #[test]
    fn test_parquet_streaming_reader_empty_data() {
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = ListBuilder::new(Float64Builder::new());
        let array = Arc::new(builder.finish()) as ArrayRef;
        let file = write_test_parquet(schema, vec![array]);

        let mut reader =
            ParquetStreamingReader::new(file.path(), None, NullHandling::FillZero).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("no data") || err_msg.contains("No data"));
    }

    // --- NullHandling tests ---

    fn write_list_parquet_with_nulls() -> TempTestFile {
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = ListBuilder::new(Float64Builder::new());
        builder.values().append_value(1.0);
        builder.values().append_null();
        builder.append(true);
        builder.values().append_value(3.0);
        builder.values().append_value(4.0);
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        write_test_parquet(schema, vec![array])
    }

    fn write_fixed_size_list_parquet_with_nulls() -> TempTestFile {
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::FixedSizeList(item_field.clone(), 2), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), 2);
        builder.values().append_value(1.0);
        builder.values().append_null();
        builder.append(true);
        builder.values().append_value(3.0);
        builder.values().append_value(4.0);
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        write_test_parquet(schema, vec![array])
    }

    #[test]
    fn test_parquet_reader_list_f64_null_fill_zero() {
        let file = write_list_parquet_with_nulls();
        let mut reader = ParquetReader::new(file.path(), None, NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 3.0, 4.0]);
        assert_eq!(num_samples, 2);
        assert_eq!(sample_size, 2);
    }

    #[test]
    fn test_parquet_reader_list_f64_null_reject() {
        let file = write_list_parquet_with_nulls();
        let mut reader = ParquetReader::new(file.path(), None, NullHandling::Reject).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Null value"));
    }

    #[test]
    fn test_parquet_reader_fixed_size_list_null_fill_zero() {
        let file = write_fixed_size_list_parquet_with_nulls();
        let mut reader = ParquetReader::new(file.path(), None, NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 3.0, 4.0]);
        assert_eq!(num_samples, 2);
        assert_eq!(sample_size, 2);
    }

    #[test]
    fn test_parquet_reader_fixed_size_list_null_reject() {
        let file = write_fixed_size_list_parquet_with_nulls();
        let mut reader = ParquetReader::new(file.path(), None, NullHandling::Reject).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Null value"));
    }

    // --- ParquetStreamingReader error-path tests ---

    #[test]
    fn test_parquet_streaming_reader_missing_file() {
        let file = TempTestFile::new();
        let path = file.path().to_path_buf();
        drop(file);
        let result = ParquetStreamingReader::new(&path, None, NullHandling::FillZero);
        assert!(matches!(result, Err(MahoutError::Io(_))));
    }

    #[test]
    fn test_parquet_streaming_reader_bad_schema() {
        let schema = Arc::new(Schema::new(vec![Field::new("bad", DataType::Int32, false)]));
        let array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let file = write_test_parquet(schema, vec![array]);

        let result = ParquetStreamingReader::new(file.path(), None, NullHandling::FillZero);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Expected"));
    }

    #[test]
    fn test_parquet_streaming_reader_too_many_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col1", DataType::Int32, false),
            Field::new("col2", DataType::Int32, false),
        ]));
        let arr1 = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let arr2 = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let file = write_test_parquet(schema, vec![arr1, arr2]);

        let result = ParquetStreamingReader::new(file.path(), None, NullHandling::FillZero);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Expected exactly one column"));
    }

    #[test]
    fn test_parquet_streaming_reader_bad_list_type() {
        let item_field = Arc::new(Field::new("item", DataType::Int32, true));
        let list_field = Field::new("list_col", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = arrow::array::ListBuilder::new(arrow::array::Int32Builder::new());
        builder.values().append_value(1);
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        let file = write_test_parquet(schema, vec![array]);

        let result = ParquetStreamingReader::new(file.path(), None, NullHandling::FillZero);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Expected List<Float64>"));
    }
}
