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

use arrow::array::{Array, Float32Array, Float64Array, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, Float32Type, Float64Type};

use crate::error::{MahoutError, Result};

/// Maps a Rust float primitive to its Arrow array type.
///
/// `pub(crate)` seals `FloatElem`: external callers cannot implement `ArrowPrimitive`
/// and therefore cannot implement `FloatElem` for new types.
pub(crate) trait ArrowPrimitive {
    type ArrowType: ArrowPrimitiveType<Native = Self>;
}

impl ArrowPrimitive for f32 {
    type ArrowType = Float32Type;
}

impl ArrowPrimitive for f64 {
    type ArrowType = Float64Type;
}

/// Scalar element type for [`DataReader`] output (`f32` or `f64` only).
///
/// Sealed by the `pub(crate) ArrowPrimitive` supertrait — no external implementations
/// are possible. Keeps f32 file data as `Vec<f32>` end-to-end; today most readers
/// use the default `T = f64`.
#[allow(private_bounds)]
pub trait FloatElem: ArrowPrimitive + Copy + Default + Send + Sync + 'static {}

impl FloatElem for f32 {}
impl FloatElem for f64 {}

/// Policy for handling null values in float arrays.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NullHandling {
    /// Replace nulls with 0.0 (backward-compatible default).
    #[default]
    FillZero,
    /// Return an error when a null is encountered.
    Reject,
}

/// Append values from a primitive array into `output`, applying the given null policy.
///
/// When there are no nulls the fast path copies the underlying buffer directly.
pub(crate) fn handle_primitive_nulls<P: ArrowPrimitiveType>(
    output: &mut Vec<P::Native>,
    array: &PrimitiveArray<P>,
    null_handling: NullHandling,
) -> Result<()>
where
    P::Native: Default,
{
    if array.null_count() == 0 {
        output.extend_from_slice(array.values());
    } else {
        match null_handling {
            NullHandling::FillZero => {
                output.extend(array.iter().map(|opt| opt.unwrap_or_default()));
            }
            NullHandling::Reject => {
                return Err(MahoutError::InvalidInput(format!(
                    "Null value encountered in {:?} array. \
                     Use NullHandling::FillZero to replace nulls with 0.0, \
                     or clean the data at the source.",
                    P::DATA_TYPE,
                )));
            }
        }
    }
    Ok(())
}

/// Append values from a `Float64Array` into `output`, applying the given null policy.
///
/// When there are no nulls the fast path copies the underlying buffer directly.
pub fn handle_float64_nulls(
    output: &mut Vec<f64>,
    float_array: &Float64Array,
    null_handling: NullHandling,
) -> Result<()> {
    handle_primitive_nulls::<Float64Type>(output, float_array, null_handling)
}

/// Append values from a `Float32Array` into `output`, applying the given null policy.
///
/// When there are no nulls the fast path copies the underlying buffer directly.
pub fn handle_float32_nulls(
    output: &mut Vec<f32>,
    float_array: &Float32Array,
    null_handling: NullHandling,
) -> Result<()> {
    handle_primitive_nulls::<Float32Type>(output, float_array, null_handling)
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
///
/// Parameterised by [`FloatElem`] (`T` defaults to `f64` for existing readers).
pub trait DataReader<T: FloatElem = f64> {
    /// Read all data from the source.
    ///
    /// Returns a tuple of:
    /// - `Vec<T>`: Flattened batch data (all samples concatenated)
    /// - `usize`: Number of samples
    /// - `usize`: Sample size (elements per sample)
    fn read_batch(&mut self) -> Result<(Vec<T>, usize, usize)>;

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
pub trait StreamingDataReader<T: FloatElem = f64>: DataReader<T> {
    /// Read a chunk of data into the provided buffer.
    ///
    /// Returns the number of elements written to the buffer.
    /// Returns 0 when no more data is available.
    ///
    /// The implementation should respect sample boundaries - only complete
    /// samples should be written to avoid splitting samples across chunks.
    fn read_chunk(&mut self, buffer: &mut [T]) -> Result<usize>;

    /// Get the total number of rows/samples in the data source.
    ///
    /// This is useful for progress tracking and memory pre-allocation.
    fn total_rows(&self) -> usize;
}
