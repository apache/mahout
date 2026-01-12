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
