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

//! Generic data reader interface for multiple input formats.
//!
//! This module provides a trait-based architecture for reading quantum data
//! from various sources (Parquet, Arrow IPC, NumPy, PyTorch, etc.) in a
//! unified way without sacrificing performance or memory efficiency.
//!
//! # Architecture
//!
//! The reader system is based on two main traits:
//!
//! - [`DataReader`]: Basic interface for batch reading
//! - [`StreamingDataReader`]: Extended interface for chunk-by-chunk streaming
//!
//! # Example: Adding a New Format
//!
//! To add support for a new format (e.g., NumPy):
//!
//! ```rust,ignore
//! use qdp_core::reader::{DataReader, Result};
//!
//! pub struct NumpyReader {
//!     // format-specific fields
//! }
//!
//! impl DataReader for NumpyReader {
//!     fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
//!         // implementation
//!     }
//! }
//! ```

use arrow::array::{Array, Float64Array};

use crate::error::Result;

/// Policy for handling null values in Float64 arrays.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NullHandling {
    /// Replace nulls with 0.0 (backward-compatible default).
    #[default]
    FillZero,
    /// Return an error when a null is encountered.
    Reject,
}

/// Append values from a `Float64Array` into `output`, applying the given null policy.
///
/// When there are no nulls the fast path copies the underlying buffer directly.
pub fn handle_float64_nulls(
    output: &mut Vec<f64>,
    float_array: &Float64Array,
    null_handling: NullHandling,
) -> crate::error::Result<()> {
    if float_array.null_count() == 0 {
        output.extend_from_slice(float_array.values());
    } else {
        match null_handling {
            NullHandling::FillZero => {
                output.extend(float_array.iter().map(|opt| opt.unwrap_or(0.0)));
            }
            NullHandling::Reject => {
                return Err(crate::error::MahoutError::InvalidInput(
                    "Null value encountered in Float64Array. \
                     Use NullHandling::FillZero to replace nulls with 0.0, \
                     or clean the data at the source."
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

/// Generic data reader interface for batch quantum data.
///
/// Implementations should read data in the format:
/// - Flattened batch data (all samples concatenated)
/// - Number of samples
/// - Sample size (elements per sample)
///
/// This interface enables zero-copy streaming where possible and maintains
/// memory efficiency for large datasets.
pub trait DataReader {
    /// Read all data from the source.
    ///
    /// Returns a tuple of:
    /// - `Vec<f64>`: Flattened batch data (all samples concatenated)
    /// - `usize`: Number of samples
    /// - `usize`: Sample size (elements per sample)
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)>;

    /// Get the sample size if known before reading.
    ///
    /// This is useful for pre-allocating buffers. Returns `None` if
    /// the sample size is not known until data is read.
    fn get_sample_size(&self) -> Option<usize> {
        None
    }

    /// Get the total number of samples if known before reading.
    ///
    /// Returns `None` if the count is not known until data is read.
    fn get_num_samples(&self) -> Option<usize> {
        None
    }
}

/// Streaming data reader interface for large datasets.
///
/// This trait enables chunk-by-chunk reading for datasets that don't fit
/// in memory, maintaining constant memory usage regardless of file size.
pub trait StreamingDataReader: DataReader {
    /// Read a chunk of data into the provided buffer.
    ///
    /// Returns the number of elements written to the buffer.
    /// Returns 0 when no more data is available.
    ///
    /// The implementation should respect sample boundaries - only complete
    /// samples should be written to avoid splitting samples across chunks.
    fn read_chunk(&mut self, buffer: &mut [f64]) -> Result<usize>;

    /// Get the total number of rows/samples in the data source.
    ///
    /// This is useful for progress tracking and memory pre-allocation.
    fn total_rows(&self) -> usize;
}
