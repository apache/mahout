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

use arrow::array::{ArrayRef, Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;

use crate::error::{MahoutError, Result};
use crate::reader::{NullHandling, handle_float64_nulls};

/// Converts an Arrow Float64Array to Vec<f64>.
pub fn arrow_to_vec(array: &Float64Array, null_handling: NullHandling) -> Result<Vec<f64>> {
    let mut result = Vec::with_capacity(array.len());
    handle_float64_nulls(&mut result, array, null_handling)?;
    Ok(result)
}

/// Flattens multiple Arrow Float64Arrays into a single Vec<f64>.
pub fn arrow_to_vec_chunked(
    arrays: &[Float64Array],
    null_handling: NullHandling,
) -> Result<Vec<f64>> {
    let total_len: usize = arrays.iter().map(|a| a.len()).sum();
    let mut result = Vec::with_capacity(total_len);

    for array in arrays {
        handle_float64_nulls(&mut result, array, null_handling)?;
    }

    Ok(result)
}

/// Reads Float64 data from a Parquet file.
///
/// Expects a single Float64 column. For zero-copy access, use [`read_parquet_to_arrow`].
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Vec<f64>> {
    let chunks = read_parquet_to_arrow(path)?;
    arrow_to_vec_chunked(&chunks, NullHandling::FillZero)
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

    let batch = RecordBatch::try_new(schema.clone(), vec![array_ref])
        .map_err(|e| MahoutError::Io(format!("Failed to create RecordBatch: {}", e)))?;

    let file = File::create(path.as_ref()).map_err(|e| MahoutError::IoWithSource {
        message: format!("Failed to create Parquet file: {}", e),
        source: e,
    })?;

    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| MahoutError::Io(format!("Failed to create Parquet writer: {}", e)))?;

    writer
        .write(&batch)
        .map_err(|e| MahoutError::Io(format!("Failed to write Parquet batch: {}", e)))?;

    writer
        .close()
        .map_err(|e| MahoutError::Io(format!("Failed to close Parquet writer: {}", e)))?;

    Ok(())
}

/// Reads a Parquet file as Arrow Float64Arrays.
///
/// Returns one array per row group for zero-copy access.
pub fn read_parquet_to_arrow<P: AsRef<Path>>(path: P) -> Result<Vec<Float64Array>> {
    let file = File::open(path.as_ref()).map_err(|e| MahoutError::IoWithSource {
        message: format!("Failed to open Parquet file: {}", e),
        source: e,
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| MahoutError::Io(format!("Failed to create Parquet reader: {}", e)))?;

    let reader = builder
        .build()
        .map_err(|e| MahoutError::Io(format!("Failed to build Parquet reader: {}", e)))?;

    let mut arrays = Vec::new();

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| MahoutError::Io(format!("Failed to read Parquet batch: {}", e)))?;

        if batch.num_columns() == 0 {
            return Err(MahoutError::Io("Parquet file has no columns".to_string()));
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
            .ok_or_else(|| MahoutError::Io("Failed to downcast to Float64Array".to_string()))?
            .clone();

        arrays.push(float_array);
    }

    if arrays.is_empty() {
        return Err(MahoutError::Io("Parquet file contains no data".to_string()));
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
    let batch = RecordBatch::try_new(schema.clone(), vec![array_ref])
        .map_err(|e| MahoutError::Io(format!("Failed to create RecordBatch: {}", e)))?;

    let file = File::create(path.as_ref()).map_err(|e| MahoutError::IoWithSource {
        message: format!("Failed to create Parquet file: {}", e),
        source: e,
    })?;

    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| MahoutError::Io(format!("Failed to create Parquet writer: {}", e)))?;

    writer
        .write(&batch)
        .map_err(|e| MahoutError::Io(format!("Failed to write Parquet batch: {}", e)))?;

    writer
        .close()
        .map_err(|e| MahoutError::Io(format!("Failed to close Parquet writer: {}", e)))?;

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
    use crate::reader::DataReader;
    let mut reader = crate::readers::ParquetReader::new(path, None, NullHandling::FillZero)?;
    reader.read_batch()
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
    use crate::reader::DataReader;
    let mut reader = crate::readers::ArrowIPCReader::new(path, NullHandling::FillZero)?;
    reader.read_batch()
}

/// Reads batch data from a NumPy .npy file.
///
/// Expects a 2D array with shape `[num_samples, sample_size]` and dtype `float64`.
/// Returns flattened data suitable for batch encoding.
///
/// # Returns
/// Tuple of `(flattened_data, num_samples, sample_size)`
///
/// # Example
/// ```rust,ignore
/// let (data, num_samples, sample_size) = read_numpy_batch("quantum_states.npy")?;
/// ```
pub fn read_numpy_batch<P: AsRef<Path>>(path: P) -> Result<(Vec<f64>, usize, usize)> {
    use crate::reader::DataReader;
    let mut reader = crate::readers::NumpyReader::new(path)?;
    reader.read_batch()
}

/// Reads batch data from a PyTorch .pt/.pth file.
///
/// Expects a 1D or 2D tensor saved with `torch.save`.
/// Returns flattened data suitable for batch encoding.
/// Requires the `pytorch` feature to be enabled.
///
/// # Returns
/// Tuple of `(flattened_data, num_samples, sample_size)`
pub fn read_torch_batch<P: AsRef<Path>>(path: P) -> Result<(Vec<f64>, usize, usize)> {
    use crate::reader::DataReader;
    let mut reader = crate::readers::TorchReader::new(path)?;
    reader.read_batch()
}

/// Streaming Parquet reader for List<Float64> and FixedSizeList<Float64> columns
///
/// Reads Parquet files in chunks without loading entire file into memory.
/// Supports efficient streaming for large files via Producer-Consumer pattern.
///
/// This is a type alias for backward compatibility. Use [`crate::readers::ParquetStreamingReader`] directly.
pub type ParquetBlockReader = crate::readers::ParquetStreamingReader;

/// Reads batch data from a TensorFlow TensorProto file.
///
/// Supports Float64 tensors with shape [batch_size, feature_size] or [n].
/// Prefers tensor_content for efficient parsing, but still requires one copy to Vec<f64>.
///
/// # Byte Order
/// Assumes little-endian byte order (standard on x86_64).
///
/// # Returns
/// Tuple of `(flattened_data, num_samples, sample_size)`
///
/// # TODO
/// Add OOM protection for very large files
pub fn read_tensorflow_batch<P: AsRef<Path>>(path: P) -> Result<(Vec<f64>, usize, usize)> {
    use crate::reader::DataReader;
    let mut reader = crate::readers::TensorFlowReader::new(path)?;
    reader.read_batch()
}
