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
use std::marker::PhantomData;
use std::path::Path;

use arrow::array::{Array, FixedSizeListArray, Float32Array, Float64Array, ListArray};
use arrow::compute;
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::{MahoutError, Result};
use crate::reader::{
    DataReader, FloatElem, NullHandling, StreamingDataReader, handle_float32_nulls,
    handle_float64_nulls,
};

// ---------------------------------------------------------------------------
// Internal sealed helper trait
// ---------------------------------------------------------------------------

/// Zero-copy or Arrow-cast extraction from an Arrow array into an output `Vec<Self>`.
///
/// Implemented for `f32` and `f64` only:
/// - same dtype → `extend_from_slice` directly from the Arrow buffer (zero alloc)
/// - cross dtype → `arrow::compute::cast` first, then `extend_from_slice`
///   - f32 → f64: exact, NaN preserved
///   - f64 → f32: values outside f32 range become ±Inf, NaN preserved
pub(crate) trait ArrowPrimitive: FloatElem {
    fn extend_from_arrow_array(
        output: &mut Vec<Self>,
        array: &dyn Array,
        null_handling: NullHandling,
    ) -> Result<()>;

    fn collect_from_arrow_array(
        array: &dyn Array,
        null_handling: NullHandling,
    ) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        Self::extend_from_arrow_array(&mut out, array, null_handling)?;
        Ok(out)
    }
}

impl ArrowPrimitive for f64 {
    fn extend_from_arrow_array(
        output: &mut Vec<f64>,
        array: &dyn Array,
        null_handling: NullHandling,
    ) -> Result<()> {
        match array.data_type() {
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| MahoutError::Io("Float64 downcast failed".to_string()))?;
                handle_float64_nulls(output, arr, null_handling)?;
            }
            DataType::Float32 => {
                let casted = compute::cast(array, &DataType::Float64)
                    .map_err(|e| MahoutError::Io(format!("Arrow cast f32→f64: {e}")))?;
                let arr = casted
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| MahoutError::Io("Cast to Float64 failed".to_string()))?;
                handle_float64_nulls(output, arr, null_handling)?;
            }
            other => {
                return Err(MahoutError::InvalidInput(format!(
                    "Expected Float32 or Float64 values, got {other:?}"
                )));
            }
        }
        Ok(())
    }
}

impl ArrowPrimitive for f32 {
    fn extend_from_arrow_array(
        output: &mut Vec<f32>,
        array: &dyn Array,
        null_handling: NullHandling,
    ) -> Result<()> {
        match array.data_type() {
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| MahoutError::Io("Float32 downcast failed".to_string()))?;
                handle_float32_nulls(output, arr, null_handling)?;
            }
            DataType::Float64 => {
                // f64 → f32: values outside f32 range become ±Inf; NaN is preserved
                let casted = compute::cast(array, &DataType::Float32)
                    .map_err(|e| MahoutError::Io(format!("Arrow cast f64→f32: {e}")))?;
                let arr = casted
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| MahoutError::Io("Cast to Float32 failed".to_string()))?;
                handle_float32_nulls(output, arr, null_handling)?;
            }
            other => {
                return Err(MahoutError::InvalidInput(format!(
                    "Expected Float32 or Float64 values, got {other:?}"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared schema validator
// ---------------------------------------------------------------------------

fn validate_float_list_schema(field: &arrow::datatypes::Field) -> Result<()> {
    match field.data_type() {
        DataType::List(child_field) => {
            if !matches!(
                child_field.data_type(),
                DataType::Float32 | DataType::Float64
            ) {
                return Err(MahoutError::InvalidInput(format!(
                    "Expected List<Float32> or List<Float64> column, got List<{:?}>",
                    child_field.data_type()
                )));
            }
        }
        DataType::FixedSizeList(child_field, _) => {
            if !matches!(
                child_field.data_type(),
                DataType::Float32 | DataType::Float64
            ) {
                return Err(MahoutError::InvalidInput(format!(
                    "Expected FixedSizeList<Float32> or FixedSizeList<Float64> column, got FixedSizeList<{:?}>",
                    child_field.data_type()
                )));
            }
        }
        _ => {
            return Err(MahoutError::InvalidInput(format!(
                "Expected List<Float32/Float64> or FixedSizeList<Float32/Float64> column, got {:?}",
                field.data_type()
            )));
        }
    }
    Ok(())
}

fn validate_float_list_or_scalar_schema(field: &arrow::datatypes::Field) -> Result<()> {
    match field.data_type() {
        DataType::Float32 | DataType::Float64 => Ok(()),
        _ => validate_float_list_schema(field),
    }
}

// ---------------------------------------------------------------------------
// ParquetReader<T>
// ---------------------------------------------------------------------------

/// Reader for Parquet files containing `List<Float32/Float64>` or
/// `FixedSizeList<Float32/Float64>` columns.
///
/// Generic over `T` (`f32` or `f64`):
/// - same dtype as the file → zero-copy path via `extend_from_slice`
/// - different dtype → `arrow::compute::cast` (f64→f32: overflow → ±Inf; NaN preserved)
pub struct ParquetReader<T: FloatElem = f64> {
    reader: Option<parquet::arrow::arrow_reader::ParquetRecordBatchReader>,
    sample_size: Option<usize>,
    total_rows: usize,
    null_handling: NullHandling,
    _phantom: PhantomData<T>,
}

#[allow(private_bounds)]
impl<T: FloatElem + ArrowPrimitive> ParquetReader<T> {
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

        validate_float_list_schema(&schema.fields()[0])?;

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
            _phantom: PhantomData,
        })
    }
}

impl<T: FloatElem + ArrowPrimitive> DataReader<T> for ParquetReader<T> {
    fn read_batch(&mut self) -> Result<(Vec<T>, usize, usize)> {
        let reader = self
            .reader
            .take()
            .ok_or_else(|| MahoutError::InvalidInput("Reader already consumed".to_string()))?;

        let mut all_data: Vec<T> = Vec::new();
        let mut num_samples = 0;
        let mut sample_size: Option<usize> = None;

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
                        let current_size = value_array.len();

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

                        T::extend_from_arrow_array(
                            &mut all_data,
                            &*value_array,
                            self.null_handling,
                        )?;
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
                    T::extend_from_arrow_array(&mut all_data, values, self.null_handling)?;
                    num_samples += list_array.len();
                }
                _ => {
                    return Err(MahoutError::Io(format!(
                        "Expected List<Float32/Float64> or FixedSizeList<Float32/Float64>, got {:?}",
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

// ---------------------------------------------------------------------------
// ParquetStreamingReader<T>
// ---------------------------------------------------------------------------

/// Streaming Parquet reader for `List<Float32/Float64>` and
/// `FixedSizeList<Float32/Float64>` columns.
///
/// Reads Parquet files in chunks without loading the entire file into memory.
/// Supports efficient streaming for large files via the Producer-Consumer pattern.
pub struct ParquetStreamingReader<T: FloatElem = f64> {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    sample_size: Option<usize>,
    leftover_data: Vec<T>,
    leftover_cursor: usize,
    pub total_rows: usize,
    null_handling: NullHandling,
    _phantom: PhantomData<T>,
}

#[allow(private_bounds)]
impl<T: FloatElem + ArrowPrimitive> ParquetStreamingReader<T> {
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

        validate_float_list_or_scalar_schema(&schema.fields()[0])?;

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
            _phantom: PhantomData,
        })
    }

    /// Get the sample size (number of elements per sample).
    pub fn get_sample_size(&self) -> Option<usize> {
        self.sample_size
    }
}

impl<T: FloatElem + ArrowPrimitive> DataReader<T> for ParquetStreamingReader<T> {
    fn read_batch(&mut self) -> Result<(Vec<T>, usize, usize)> {
        let mut all_data = Vec::new();
        let mut num_samples = 0;

        loop {
            let mut buffer = vec![T::default(); 1024 * 1024];
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

impl<T: FloatElem + ArrowPrimitive> StreamingDataReader<T> for ParquetStreamingReader<T> {
    fn read_chunk(&mut self, buffer: &mut [T]) -> Result<usize> {
        let mut written = 0;
        let buf_cap = buffer.len();
        let calc_limit = |ss: usize| -> usize {
            buf_cap
                .checked_div(ss)
                .map(|chunks| chunks * ss)
                .unwrap_or(buf_cap)
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

                            let mut batch_values: Vec<T> = Vec::new();
                            let mut current_sample_size = None;
                            for i in 0..list_array.len() {
                                let value_array = list_array.value(i);
                                if i == 0 {
                                    current_sample_size = Some(value_array.len());
                                }
                                T::extend_from_arrow_array(
                                    &mut batch_values,
                                    &*value_array,
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
                            let batch_values =
                                T::collect_from_arrow_array(values, self.null_handling)?;

                            (current_sample_size, batch_values)
                        }
                        DataType::Float32 | DataType::Float64 => {
                            // Scalar float for basis encoding (one index per sample)
                            if column.is_empty() {
                                continue;
                            }
                            let current_sample_size = 1;
                            let batch_values =
                                T::collect_from_arrow_array(&**column, self.null_handling)?;
                            (current_sample_size, batch_values)
                        }
                        _ => {
                            return Err(MahoutError::Io(format!(
                                "Expected Float32/Float64, List<Float32/Float64>, or FixedSizeList<Float32/Float64>, got {:?}",
                                column.data_type()
                            )));
                        }
                    };

                    match self.sample_size {
                        Some(expected_size) if current_sample_size != expected_size => {
                            return Err(MahoutError::InvalidInput(format!(
                                "Inconsistent sample sizes: expected {}, got {}",
                                expected_size, current_sample_size
                            )));
                        }
                        None => {
                            self.sample_size = Some(current_sample_size);
                            limit = calc_limit(current_sample_size);
                        }
                        _ => {}
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

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
        let result = ParquetReader::<f64>::new(&path, None, NullHandling::FillZero);
        assert!(matches!(result, Err(MahoutError::Io(_))));
    }

    #[test]
    fn test_parquet_reader_bad_schema() {
        let schema = Arc::new(Schema::new(vec![Field::new("bad", DataType::Int32, false)]));
        let array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let file = write_test_parquet(schema, vec![array]);

        let result = ParquetReader::<f64>::new(file.path(), None, NullHandling::FillZero);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(
            err_msg.contains("Expected List")
                || err_msg.contains("Expected FixedSizeList")
                || err_msg.contains("Float32/Float64")
        );
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

        let result = ParquetReader::<f64>::new(file.path(), None, NullHandling::FillZero);
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

        let result = ParquetReader::<f64>::new(file.path(), None, NullHandling::FillZero);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Expected List<Float32> or List<Float64>"));
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

        let mut reader =
            ParquetReader::<f64>::new(file.path(), None, NullHandling::FillZero).unwrap();
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

        let mut reader =
            ParquetReader::<f64>::new(file.path(), None, NullHandling::FillZero).unwrap();
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

        let mut reader =
            ParquetReader::<f64>::new(file.path(), None, NullHandling::FillZero).unwrap();
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
        use arrow::array::Float64Builder;
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
            ParquetStreamingReader::<f64>::new(file.path(), None, NullHandling::FillZero).unwrap();
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
            ParquetStreamingReader::<f64>::new(file.path(), Some(1), NullHandling::FillZero)
                .unwrap();
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

        let mut reader =
            ParquetReader::<f64>::new(file.path(), None, NullHandling::FillZero).unwrap();
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
            ParquetStreamingReader::<f64>::new(file.path(), None, NullHandling::FillZero).unwrap();
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
        let mut reader =
            ParquetReader::<f64>::new(file.path(), None, NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 3.0, 4.0]);
        assert_eq!(num_samples, 2);
        assert_eq!(sample_size, 2);
    }

    #[test]
    fn test_parquet_reader_list_f64_null_reject() {
        let file = write_list_parquet_with_nulls();
        let mut reader =
            ParquetReader::<f64>::new(file.path(), None, NullHandling::Reject).unwrap();
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
        let mut reader =
            ParquetReader::<f64>::new(file.path(), None, NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 3.0, 4.0]);
        assert_eq!(num_samples, 2);
        assert_eq!(sample_size, 2);
    }

    #[test]
    fn test_parquet_reader_fixed_size_list_null_reject() {
        let file = write_fixed_size_list_parquet_with_nulls();
        let mut reader =
            ParquetReader::<f64>::new(file.path(), None, NullHandling::Reject).unwrap();
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
        let result = ParquetStreamingReader::<f64>::new(&path, None, NullHandling::FillZero);
        assert!(matches!(result, Err(MahoutError::Io(_))));
    }

    #[test]
    fn test_parquet_streaming_reader_bad_schema() {
        let schema = Arc::new(Schema::new(vec![Field::new("bad", DataType::Int32, false)]));
        let array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let file = write_test_parquet(schema, vec![array]);

        let result = ParquetStreamingReader::<f64>::new(file.path(), None, NullHandling::FillZero);
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

        let result = ParquetStreamingReader::<f64>::new(file.path(), None, NullHandling::FillZero);
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

        let result = ParquetStreamingReader::<f64>::new(file.path(), None, NullHandling::FillZero);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!(),
        };
        assert!(err_msg.contains("Expected List<Float32> or List<Float64>"));
    }
}
