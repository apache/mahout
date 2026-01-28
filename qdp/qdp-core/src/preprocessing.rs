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

use crate::error::{MahoutError, Result};
use crate::gpu::encodings::validate_qubit_count;
use rayon::prelude::*;

/// Shared CPU-based pre-processing pipeline for quantum encoding.
///
/// Centralizes validation, normalization, and data preparation steps
/// to ensure consistency across different encoding strategies and backends.
pub struct Preprocessor;

impl Preprocessor {
    /// Validates standard quantum input constraints.
    ///
    /// Checks:
    /// - Qubit count within practical limits (1-MAX_QUBITS)
    /// - Data availability
    /// - Data length against state vector size
    pub fn validate_input(host_data: &[f64], num_qubits: usize) -> Result<()> {
        // Validate qubits using shared validation function (max MAX_QUBITS = 16GB GPU memory)
        validate_qubit_count(num_qubits)?;

        // Validate input data
        if host_data.is_empty() {
            return Err(MahoutError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        let state_len = 1 << num_qubits;
        if host_data.len() > state_len {
            return Err(MahoutError::InvalidInput(format!(
                "Input data length {} exceeds state vector size {}",
                host_data.len(),
                state_len
            )));
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
            return Err(MahoutError::InvalidInput(
                "Input data has zero norm".to_string(),
            ));
        }

        if norm.is_nan() {
            return Err(MahoutError::InvalidInput(
                "Input data contains NaN (Not a Number) values".to_string(),
            ));
        }

        if norm.is_infinite() {
            return Err(MahoutError::InvalidInput(
                "Input data contains Infinity values".to_string(),
            ));
        }

        Ok(norm)
    }

    /// Validates input constraints for batch processing.
    pub fn validate_batch(
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
    ) -> Result<()> {
        if num_samples == 0 {
            return Err(MahoutError::InvalidInput(
                "num_samples must be greater than 0".to_string(),
            ));
        }

        if batch_data.len() != num_samples * sample_size {
            return Err(MahoutError::InvalidInput(format!(
                "Batch data length {} doesn't match num_samples {} * sample_size {}",
                batch_data.len(),
                num_samples,
                sample_size
            )));
        }

        // Validate qubits using shared validation function
        validate_qubit_count(num_qubits)?;

        let state_len = 1 << num_qubits;
        if sample_size > state_len {
            return Err(MahoutError::InvalidInput(format!(
                "Sample size {} exceeds state vector size {}",
                sample_size, state_len
            )));
        }

        Ok(())
    }

    /// Calculates L2 norms for a batch of samples in parallel.
    pub fn calculate_batch_l2_norms(
        batch_data: &[f64],
        _num_samples: usize,
        sample_size: usize,
    ) -> Result<Vec<f64>> {
        crate::profile_scope!("CPU::BatchL2Norm");

        // Process chunks in parallel using rayon
        batch_data
            .par_chunks(sample_size)
            .enumerate()
            .map(|(i, sample)| {
                let norm_sq: f64 = sample.iter().map(|&x| x * x).sum();
                let norm = norm_sq.sqrt();
                if norm == 0.0 {
                    return Err(MahoutError::InvalidInput(format!(
                        "Sample {} has zero norm",
                        i
                    )));
                }
                // Check result for NaN and Infinity
                if norm.is_nan() {
                    return Err(MahoutError::InvalidInput(format!(
                        "Sample {} produced NaN norm",
                        i
                    )));
                }
                if norm.is_infinite() {
                    return Err(MahoutError::InvalidInput(format!(
                        "Sample {} produced Infinity norm",
                        i
                    )));
                }
                Ok(norm)
            })
            .collect()
    }
}
