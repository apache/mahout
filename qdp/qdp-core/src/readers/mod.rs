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

//! Format-specific data reader implementations.
//!
//! This module contains concrete implementations of the [`DataReader`] and
//! [`StreamingDataReader`] traits for various file formats.
//!
//! # Fully Implemented Formats
//! - **Parquet**: [`ParquetReader`], [`ParquetStreamingReader`]
//! - **Arrow IPC**: [`ArrowIPCReader`]
//! - **NumPy**: [`NumpyReader`]
//! - **TensorFlow TensorProto**: [`TensorFlowReader`]
//! - **PyTorch**: [`TorchReader`] (feature: `pytorch`)

pub mod arrow_ipc;
pub mod numpy;
pub mod parquet;
pub mod tensorflow;
pub mod torch;

pub use arrow_ipc::ArrowIPCReader;
pub use numpy::NumpyReader;
pub use parquet::{ParquetReader, ParquetStreamingReader};
pub use tensorflow::TensorFlowReader;
pub use torch::TorchReader;

/// Error for a `List` column whose rows are all null under `NullHandling::FillZero`:
/// no row ever reveals how many zeros a null sample should become.
///
/// Shared by the Parquet and Arrow IPC readers so that both report the same
/// thing for the same file.
pub(crate) fn all_null_list_error() -> crate::MahoutError {
    crate::MahoutError::InvalidInput(
        "List column has rows but every row is null, so the sample size needed by \
         NullHandling::FillZero cannot be determined"
            .to_string(),
    )
}

/// Error for a `List` column whose first non-null row is empty.
///
/// Rows must all be the same length, so one empty row means every sample has no
/// elements. There is nothing to encode, and `NullHandling::FillZero` has no
/// width to fill a null row with.
pub(crate) fn zero_width_list_error() -> crate::MahoutError {
    crate::MahoutError::InvalidInput(
        "List column has a non-null row of length 0, so a sample would have no \
         elements"
            .to_string(),
    )
}
