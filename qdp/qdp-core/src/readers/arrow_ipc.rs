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

//! Arrow IPC format reader implementation.

use std::fs::File;
use std::path::Path;

use arrow::array::{Array, FixedSizeListArray, Float64Array, ListArray};
use arrow::datatypes::DataType;
use arrow::ipc::reader::FileReader as ArrowFileReader;

use crate::error::{MahoutError, Result};
use crate::reader::DataReader;

/// Reader for Arrow IPC files containing FixedSizeList<Float64> or List<Float64> columns.
pub struct ArrowIPCReader {
    path: std::path::PathBuf,
    read: bool,
}

impl ArrowIPCReader {
    /// Create a new Arrow IPC reader.
    ///
    /// # Arguments
    /// * `path` - Path to the Arrow IPC file (.arrow or .feather)
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Verify file exists
        match path.try_exists() {
            Ok(false) => {
                return Err(MahoutError::Io(format!(
                    "Arrow IPC file not found: {}",
                    path.display()
                )));
            }
            Err(e) => {
                return Err(MahoutError::Io(format!(
                    "Failed to check if Arrow IPC file exists at {}: {}",
                    path.display(),
                    e
                )));
            }
            Ok(true) => {}
        }

        Ok(Self {
            path: path.to_path_buf(),
            read: false,
        })
    }
}

impl DataReader for ArrowIPCReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        if self.read {
            return Err(MahoutError::InvalidInput(
                "Reader already consumed".to_string(),
            ));
        }
        self.read = true;

        let file = File::open(&self.path)
            .map_err(|e| MahoutError::Io(format!("Failed to open Arrow IPC file: {}", e)))?;

        let reader = ArrowFileReader::try_new(file, None)
            .map_err(|e| MahoutError::Io(format!("Failed to create Arrow IPC reader: {}", e)))?;

        let mut all_data = Vec::new();
        let mut num_samples = 0;
        let mut sample_size: Option<usize> = None;

        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| MahoutError::Io(format!("Failed to read Arrow batch: {}", e)))?;

            if batch.num_columns() == 0 {
                return Err(MahoutError::Io("Arrow file has no columns".to_string()));
            }

            let column = batch.column(0);

            match column.data_type() {
                DataType::FixedSizeList(_, size) => {
                    let list_array = column
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .ok_or_else(|| {
                            MahoutError::Io("Failed to downcast to FixedSizeListArray".to_string())
                        })?;

                    let current_size = *size as usize;

                    if let Some(expected) = sample_size {
                        if current_size != expected {
                            return Err(MahoutError::InvalidInput(format!(
                                "Inconsistent sample sizes: expected {}, got {}",
                                expected, current_size
                            )));
                        }
                    } else {
                        sample_size = Some(current_size);
                        let new_capacity = current_size
                            .checked_mul(batch.num_rows())
                            .expect("Capacity overflowed usize");
                        all_data.reserve(new_capacity);
                    }

                    let values = list_array.values();
                    let float_array = values
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| MahoutError::Io("Values must be Float64".to_string()))?;

                    if float_array.null_count() == 0 {
                        all_data.extend_from_slice(float_array.values());
                    } else {
                        all_data.extend(float_array.iter().map(|opt| opt.unwrap_or(0.0)));
                    }

                    num_samples += list_array.len();
                }

                DataType::List(_) => {
                    let list_array =
                        column.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                            MahoutError::Io("Failed to downcast to ListArray".to_string())
                        })?;

                    for i in 0..list_array.len() {
                        let value_array = list_array.value(i);
                        let float_array = value_array
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| {
                                MahoutError::Io("List values must be Float64".to_string())
                            })?;

                        let current_size = float_array.len();

                        if let Some(expected) = sample_size {
                            if current_size != expected {
                                return Err(MahoutError::InvalidInput(format!(
                                    "Inconsistent sample sizes: expected {}, got {}",
                                    expected, current_size
                                )));
                            }
                        } else {
                            sample_size = Some(current_size);
                            all_data.reserve(current_size * list_array.len());
                        }

                        if float_array.null_count() == 0 {
                            all_data.extend_from_slice(float_array.values());
                        } else {
                            all_data.extend(float_array.iter().map(|opt| opt.unwrap_or(0.0)));
                        }

                        num_samples += 1;
                    }
                }

                _ => {
                    return Err(MahoutError::Io(format!(
                        "Expected FixedSizeList<Float64> or List<Float64>, got {:?}",
                        column.data_type()
                    )));
                }
            }
        }

        let sample_size = sample_size
            .ok_or_else(|| MahoutError::Io("Arrow file contains no data".to_string()))?;

        Ok((all_data, num_samples, sample_size))
    }
}
