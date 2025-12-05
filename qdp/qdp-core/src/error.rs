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

use thiserror::Error;

/// Error types for Mahout QDP operations
#[derive(Error, Debug)]
pub enum MahoutError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("Kernel launch failed: {0}")]
    KernelLaunch(String),

    #[error("DLPack operation failed: {0}")]
    DLPack(String),

    #[error("I/O error: {0}")]
    Io(String),
}

/// Result type alias for Mahout operations
pub type Result<T> = std::result::Result<T, MahoutError>;
