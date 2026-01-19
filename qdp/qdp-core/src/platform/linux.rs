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

//! Linux platform-specific implementations.
//!
//! This module provides the platform entry point for Parquet streaming encoding.
//! The actual encoding implementations are in the `crate::encoding` module.

use crate::dlpack::DLManagedTensor;
use crate::{QdpEngine, Result};

/// Encode data from a Parquet file using the specified encoding method.
///
/// This is the Linux platform entry point that delegates to the encoding module.
pub(crate) fn encode_from_parquet(
    engine: &QdpEngine,
    path: &str,
    num_qubits: usize,
    encoding_method: &str,
) -> Result<*mut DLManagedTensor> {
    crate::encoding::encode_from_parquet(engine, path, num_qubits, encoding_method)
}
