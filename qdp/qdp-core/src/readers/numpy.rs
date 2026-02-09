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

//! NumPy format reader implementation.
//!
//! Provides support for reading .npy files containing 2D float64 arrays.

use std::path::Path;

use ndarray::Array2;
use ndarray_npy::ReadNpyError;

use crate::error::{MahoutError, Result};
use crate::reader::DataReader;

/// Reader for NumPy `.npy` files containing 2D float64 arrays.
///
/// # Expected Format
/// - 2D array with shape `[num_samples, sample_size]`
/// - Data type: `float64`
/// - Fortran (column-major) or C (row-major) order supported
///
/// # Example
///
/// ```rust,ignore
/// use qdp_core::reader::DataReader;
/// use qdp_core::readers::NumpyReader;
///
/// let mut reader = NumpyReader::new("data.npy").unwrap();
/// let (data, num_samples, sample_size) = reader.read_batch().unwrap();
/// println!("Read {} samples of size {}", num_samples, sample_size);
/// ```
pub struct NumpyReader {
    path: std::path::PathBuf,
    read: bool,
}

impl NumpyReader {
    /// Create a new NumPy reader.
    ///
    /// # Arguments
    /// * `path` - Path to the `.npy` file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Verify file exists
        match path.try_exists() {
            Ok(false) => {
                return Err(MahoutError::Io(format!(
                    "NumPy file not found: {}",
                    path.display()
                )));
            }
            Err(e) => {
                return Err(MahoutError::IoWithSource {
                    message: format!(
                        "Failed to check if NumPy file exists at {}: {}",
                        path.display(),
                        e
                    ),
                    source: e,
                });
            }
            Ok(true) => {}
        }

        Ok(Self {
            path: path.to_path_buf(),
            read: false,
        })
    }
}

impl DataReader for NumpyReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        if self.read {
            return Err(MahoutError::InvalidInput(
                "Reader already consumed".to_string(),
            ));
        }
        self.read = true;

        // Read the .npy file
        let array: Array2<f64> = ndarray_npy::read_npy(&self.path).map_err(|e| match e {
            ReadNpyError::Io(io_err) => MahoutError::IoWithSource {
                message: format!("Failed to read NumPy file: {}", io_err),
                source: io_err,
            },
            _ => MahoutError::InvalidInput(format!("Failed to parse NumPy file: {}", e)),
        })?;

        // Extract shape
        let shape = array.shape();
        if shape.len() != 2 {
            return Err(MahoutError::InvalidInput(format!(
                "Expected 2D array, got {}D array with shape {:?}",
                shape.len(),
                shape
            )));
        }

        let num_samples = shape[0];
        let sample_size = shape[1];

        if num_samples == 0 || sample_size == 0 {
            return Err(MahoutError::InvalidInput(format!(
                "Invalid array shape: [{}, {}]. Both dimensions must be > 0",
                num_samples, sample_size
            )));
        }

        // Flatten to Vec<f64>
        // Handle both C-contiguous (row-major) and Fortran-contiguous (column-major)
        let data = if array.is_standard_layout() {
            // C-contiguous: can use into_raw_vec_and_offset for zero-copy
            let (vec, offset) = array.into_raw_vec_and_offset();
            match offset {
                Some(off) if off > 0 => {
                    // If there's an offset, we need to copy
                    vec[off..].to_vec()
                }
                _ => vec,
            }
        } else {
            // Not C-contiguous: need to copy in row-major order
            let mut data = Vec::with_capacity(num_samples * sample_size);
            for row in array.rows() {
                data.extend(row.iter().copied());
            }
            data
        };

        Ok((data, num_samples, sample_size))
    }

    fn get_sample_size(&self) -> Option<usize> {
        // Could be determined by reading just the header
        // For now, return None as we read on demand
        None
    }

    fn get_num_samples(&self) -> Option<usize> {
        // Could be determined by reading just the header
        // For now, return None as we read on demand
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::fs;

    #[test]
    fn test_numpy_reader_basic() {
        // Create a test .npy file
        let temp_path = "/tmp/test_numpy_basic.npy";
        let num_samples = 5;
        let sample_size = 8;

        let mut data = Vec::with_capacity(num_samples * sample_size);
        for i in 0..num_samples {
            for j in 0..sample_size {
                data.push((i * sample_size + j) as f64);
            }
        }

        let array = Array2::from_shape_vec((num_samples, sample_size), data.clone()).unwrap();
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        // Read it back
        let mut reader = NumpyReader::new(temp_path).unwrap();
        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data.len(), num_samples * sample_size);
        assert_eq!(read_data, data);

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_reader_fortran_order() {
        // Create a Fortran-order (column-major) array
        let temp_path = "/tmp/test_numpy_fortran.npy";
        let num_samples = 3;
        let sample_size = 4;

        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();
        let array = Array2::from_shape_vec((num_samples, sample_size), data.clone()).unwrap();

        // Convert to Fortran order
        let array_f = array.reversed_axes();
        let array_f = array_f.as_standard_layout().reversed_axes();

        ndarray_npy::write_npy(temp_path, &array_f).unwrap();

        // Read it back
        let mut reader = NumpyReader::new(temp_path).unwrap();
        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data.len(), num_samples * sample_size);

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_reader_file_not_found() {
        let result = NumpyReader::new("/tmp/nonexistent_numpy_file_12345.npy");
        assert!(result.is_err());
    }

    #[test]
    fn test_numpy_reader_invalid_dimensions() {
        // Create a 1D array (should fail)
        let temp_path = "/tmp/test_numpy_1d.npy";
        let array = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        let mut reader = NumpyReader::new(temp_path).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_reader_already_consumed() {
        let temp_path = "/tmp/test_numpy_consumed.npy";
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        let mut reader = NumpyReader::new(temp_path).unwrap();
        let _ = reader.read_batch().unwrap();

        // Second read should fail
        let result = reader.read_batch();
        assert!(result.is_err());

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_reader_empty_dimensions() {
        // Create an array with zero dimension
        let temp_path = "/tmp/test_numpy_empty.npy";
        let array = Array2::<f64>::zeros((0, 5));
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        let mut reader = NumpyReader::new(temp_path).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }
}
