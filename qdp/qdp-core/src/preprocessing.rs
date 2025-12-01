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

use rayon::prelude::*;
use crate::error::{MahoutError, Result};

/// Shared CPU-based pre-processing pipeline for quantum encoding.
///
/// Centralizes validation, normalization, and data preparation steps
/// to ensure consistency across different encoding strategies and backends.
pub struct Preprocessor;

impl Preprocessor {
    /// Validates standard quantum input constraints.
    ///
    /// Checks:
    /// - Qubit count within practical limits (1-30)
    /// - Data availability
    /// - Data length against state vector size
    pub fn validate_input(host_data: &[f64], num_qubits: usize) -> Result<()> {
        // Validate qubits (max 30 = 16GB GPU memory)
        if num_qubits == 0 {
            return Err(MahoutError::InvalidInput(
                "Number of qubits must be at least 1".to_string()
            ));
        }
        if num_qubits > 30 {
            return Err(MahoutError::InvalidInput(
                format!("Number of qubits {} exceeds practical limit of 30", num_qubits)
            ));
        }

        // Validate input data
        if host_data.is_empty() {
            return Err(MahoutError::InvalidInput(
                "Input data cannot be empty".to_string()
            ));
        }

        let state_len = 1 << num_qubits;
        if host_data.len() > state_len {
            return Err(MahoutError::InvalidInput(
                format!("Input data length {} exceeds state vector size {}", host_data.len(), state_len)
            ));
        }

        Ok(())
    }

    /// Calculates L2 norm of the input data in parallel on the CPU.
    ///
    /// Returns error if the calculated norm is zero.
    pub fn calculate_l2_norm(host_data: &[f64]) -> Result<f64> {
        let norm = {
            crate::profile_scope!("CPU::L2Norm");
            let norm_sq: f64 = host_data.par_iter().map(|x| x * x).sum();
            norm_sq.sqrt()
        };

        if norm == 0.0 {
            return Err(MahoutError::InvalidInput("Input data has zero norm".to_string()));
        }

        Ok(norm)
    }
}
