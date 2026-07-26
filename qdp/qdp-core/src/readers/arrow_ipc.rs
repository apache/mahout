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
use crate::reader::{DataReader, NullHandling, handle_float64_nulls};

/// Reader for Arrow IPC files containing FixedSizeList<Float64> or List<Float64> columns.
pub struct ArrowIPCReader {
    path: std::path::PathBuf,
    read: bool,
    null_handling: NullHandling,
}

impl ArrowIPCReader {
    /// Create a new Arrow IPC reader.
    ///
    /// # Arguments
    /// * `path` - Path to the Arrow IPC file (.arrow or .feather)
    /// * `null_handling` - Policy for null values (defaults to `FillZero`)
    pub fn new<P: AsRef<Path>>(path: P, null_handling: NullHandling) -> Result<Self> {
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
                return Err(MahoutError::IoWithSource {
                    message: format!(
                        "Failed to check if Arrow IPC file exists at {}: {}",
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
            null_handling,
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

        let file = File::open(&self.path).map_err(|e| MahoutError::IoWithSource {
            message: format!("Failed to open Arrow IPC file: {}", e),
            source: e,
        })?;

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
                        let new_capacity =
                            current_size.checked_mul(batch.num_rows()).ok_or_else(|| {
                                MahoutError::InvalidInput(format!(
                                    "FixedSizeList capacity overflow: {} * {} would overflow usize",
                                    current_size,
                                    batch.num_rows()
                                ))
                            })?;
                        all_data.reserve(new_capacity);
                    }

                    let values = list_array.values();
                    let float_array = values
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| MahoutError::Io("Values must be Float64".to_string()))?;

                    handle_float64_nulls(&mut all_data, float_array, self.null_handling)?;

                    num_samples += list_array.len();
                }

                DataType::List(_) => {
                    let list_array =
                        column.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                            MahoutError::Io("Failed to downcast to ListArray".to_string())
                        })?;

                    // Phase 1: find sample_size from non-null rows and validate consistency.
                    for i in 0..list_array.len() {
                        if list_array.is_null(i) {
                            continue;
                        }
                        let current_size = list_array.value_length(i) as usize;

                        if let Some(expected) = sample_size {
                            if current_size != expected {
                                return Err(MahoutError::InvalidInput(format!(
                                    "Inconsistent sample sizes: expected {}, got {}",
                                    expected, current_size
                                )));
                            }
                        } else {
                            sample_size = Some(current_size);
                            let new_capacity =
                                current_size.checked_mul(list_array.len()).ok_or_else(|| {
                                    MahoutError::InvalidInput(format!(
                                        "List capacity overflow: {} * {} would overflow usize",
                                        current_size,
                                        list_array.len()
                                    ))
                                })?;
                            all_data.reserve(new_capacity);
                        }
                    }

                    // Phase 2: collect data, handling null outer rows per NullHandling policy.
                    if list_array.null_count() == 0 {
                        let values = list_array.values();
                        let float_array = values
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| MahoutError::Io("Values must be Float64".to_string()))?;
                        handle_float64_nulls(&mut all_data, float_array, self.null_handling)?;
                        num_samples += list_array.len();
                    } else {
                        for i in 0..list_array.len() {
                            if list_array.is_null(i) {
                                match self.null_handling {
                                    NullHandling::Reject => {
                                        return Err(MahoutError::InvalidInput(
                                            "Null outer row in List column. Use \
                                             NullHandling::FillZero to replace with zeros, \
                                             or clean the data at the source."
                                                .to_string(),
                                        ));
                                    }
                                    NullHandling::FillZero => {
                                        if let Some(ss) = sample_size {
                                            all_data.extend(std::iter::repeat_n(0.0_f64, ss));
                                            num_samples += 1;
                                        }
                                        // sample_size unknown: skip this null row without counting it.
                                    }
                                }
                            } else {
                                let value_array = list_array.value(i);
                                let float_array = value_array
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                    .ok_or_else(|| {
                                        MahoutError::Io("List values must be Float64".to_string())
                                    })?;
                                handle_float64_nulls(
                                    &mut all_data,
                                    float_array,
                                    self.null_handling,
                                )?;
                                num_samples += 1;
                            }
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Builder, ListBuilder, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::FileWriter as ArrowIpcFileWriter;
    use std::fs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempTestFile {
        path: std::path::PathBuf,
    }

    impl TempTestFile {
        fn new() -> Self {
            let count = TEST_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "mahout_test_arrow_ipc_{}_{}.arrow",
                std::process::id(),
                count
            ));
            Self { path }
        }

        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TempTestFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn write_test_arrow_ipc(schema: Arc<Schema>, arrays: Vec<ArrayRef>) -> TempTestFile {
        let file = TempTestFile::new();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        let os_file = fs::File::create(file.path()).unwrap();
        let mut writer = ArrowIpcFileWriter::try_new(os_file, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        file
    }

    fn write_ipc_list_with_null_outer_middle() -> TempTestFile {
        // [[1.0, 2.0], null, [3.0, 4.0]]
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = ListBuilder::new(Float64Builder::new());
        builder.values().append_slice(&[1.0, 2.0]);
        builder.append(true);
        builder.append(false); // null outer row
        builder.values().append_slice(&[3.0, 4.0]);
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        write_test_arrow_ipc(schema, vec![array])
    }

    fn write_ipc_list_with_null_outer_first() -> TempTestFile {
        // [null, [1.0, 2.0], [3.0, 4.0]] — null row at position 0
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut builder = ListBuilder::new(Float64Builder::new());
        builder.append(false); // null outer row at position 0
        builder.values().append_slice(&[1.0, 2.0]);
        builder.append(true);
        builder.values().append_slice(&[3.0, 4.0]);
        builder.append(true);
        let array = Arc::new(builder.finish()) as ArrayRef;

        write_test_arrow_ipc(schema, vec![array])
    }

    #[test]
    fn test_arrow_ipc_reader_null_outer_row_middle_fill_zero() {
        // [[1,2], null, [3,4]] with FillZero → [1,2, 0,0, 3,4]
        let file = write_ipc_list_with_null_outer_middle();
        let mut reader = ArrowIPCReader::new(file.path(), NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
        assert_eq!(num_samples, 3);
        assert_eq!(sample_size, 2);
    }

    #[test]
    fn test_arrow_ipc_reader_null_outer_row_middle_reject() {
        // [[1,2], null, [3,4]] with Reject → error
        let file = write_ipc_list_with_null_outer_middle();
        let mut reader = ArrowIPCReader::new(file.path(), NullHandling::Reject).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Null outer row"));
    }

    #[test]
    fn test_arrow_ipc_reader_null_outer_row_first_fill_zero() {
        // [null, [1,2], [3,4]] — null at row 0 must not corrupt sample_size
        let file = write_ipc_list_with_null_outer_first();
        let mut reader = ArrowIPCReader::new(file.path(), NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(num_samples, 3);
        assert_eq!(sample_size, 2);
    }

    #[test]
    fn test_arrow_ipc_reader_cross_batch_all_null_first_fill_zero() {
        // Batch 1: [null, null]   — sample_size unknown
        // Batch 2: [[1,2], [3,4]] — sample_size established here
        // All-null leading batch must not corrupt num_samples.
        let item_field = Arc::new(Field::new("item", DataType::Float64, true));
        let list_field = Field::new("data", DataType::List(item_field.clone()), true);
        let schema = Arc::new(Schema::new(vec![list_field]));

        let mut b1 = ListBuilder::new(Float64Builder::new());
        b1.append(false);
        b1.append(false);
        let batch1 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(b1.finish()) as ArrayRef]).unwrap();

        let mut b2 = ListBuilder::new(Float64Builder::new());
        b2.values().append_slice(&[1.0, 2.0]);
        b2.append(true);
        b2.values().append_slice(&[3.0, 4.0]);
        b2.append(true);
        let batch2 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(b2.finish()) as ArrayRef]).unwrap();

        let file = TempTestFile::new();
        {
            let os_file = fs::File::create(file.path()).unwrap();
            let mut writer = ArrowIpcFileWriter::try_new(os_file, &schema).unwrap();
            writer.write(&batch1).unwrap();
            writer.write(&batch2).unwrap();
            writer.finish().unwrap();
        }

        let mut reader = ArrowIPCReader::new(file.path(), NullHandling::FillZero).unwrap();
        let (data, num_samples, sample_size) = reader.read_batch().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(num_samples, 2);
        assert_eq!(sample_size, 2);
    }
}
