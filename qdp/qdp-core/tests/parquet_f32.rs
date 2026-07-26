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

//! Acceptance tests for ParquetReader<f32> — issue #1340.

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{
    ArrayRef, FixedSizeListBuilder, Float32Builder, Float64Builder, ListBuilder, RecordBatch,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use qdp_core::reader::{DataReader, NullHandling};
use qdp_core::readers::parquet::{ParquetReader, ParquetStreamingReader};

static FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempFile(std::path::PathBuf);

impl TempFile {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn write_list_parquet(schema: Arc<Schema>, arrays: Vec<ArrayRef>) -> TempFile {
    let n = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "mahout_parquet_f32_{}_{}.parquet",
        std::process::id(),
        n,
    ));
    let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
    let file = fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    TempFile(path)
}

// ---------------------------------------------------------------------------
// Acceptance test 1: f32 column read as f32 (zero-copy path)
// ---------------------------------------------------------------------------

/// ParquetReader::<f32> on a List<Float32> file → values come back as Vec<f32>,
/// no precision loss, correct count.
#[test]
fn test_f32_column_read_as_f32() {
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let list_field = Field::new("data", DataType::List(item_field), true);
    let schema = Arc::new(Schema::new(vec![list_field]));

    let mut builder = ListBuilder::new(Float32Builder::new());
    builder.values().append_slice(&[1.0_f32, 2.5_f32, 3.75_f32]);
    builder.append(true);
    builder.values().append_slice(&[4.0_f32, 5.5_f32, 6.25_f32]);
    builder.append(true);
    let array = Arc::new(builder.finish()) as ArrayRef;

    let tmp = write_list_parquet(schema, vec![array]);

    let mut reader = ParquetReader::<f32>::new(tmp.path(), None, NullHandling::FillZero).unwrap();
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();

    assert_eq!(num_samples, 2);
    assert_eq!(sample_size, 3);
    assert_eq!(data, vec![1.0_f32, 2.5, 3.75, 4.0, 5.5, 6.25]);
}

// ---------------------------------------------------------------------------
// Acceptance test 2: f64 column cast to f32
// ---------------------------------------------------------------------------

/// ParquetReader::<f32> on a List<Float64> file → Arrow cast applied.
/// Normal values: cast is precise within f32 range.
/// Overflow (f64 > f32::MAX): → +Inf (Arrow safe cast behaviour).
/// NaN: preserved.
#[test]
fn test_f64_column_cast_to_f32() {
    let item_field = Arc::new(Field::new("item", DataType::Float64, true));
    let list_field = Field::new("data", DataType::List(item_field), true);
    let schema = Arc::new(Schema::new(vec![list_field]));

    let overflow = f64::from(f32::MAX) * 2.0; // overflows f32 → +Inf after cast
    let nan = f64::NAN;

    let mut builder = ListBuilder::new(Float64Builder::new());
    builder.values().append_slice(&[1.0_f64, -2.0_f64]);
    builder.append(true);
    builder.values().append_value(overflow);
    builder.values().append_value(nan);
    builder.append(true);
    let array = Arc::new(builder.finish()) as ArrayRef;

    let tmp = write_list_parquet(schema, vec![array]);

    let mut reader = ParquetReader::<f32>::new(tmp.path(), None, NullHandling::FillZero).unwrap();
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();

    assert_eq!(num_samples, 2);
    assert_eq!(sample_size, 2);
    assert_eq!(data[0], 1.0_f32);
    assert_eq!(data[1], -2.0_f32);
    assert!(
        data[2].is_infinite() && data[2] > 0.0,
        "expected +Inf, got {}",
        data[2]
    );
    assert!(data[3].is_nan(), "expected NaN, got {}", data[3]);
}

// ---------------------------------------------------------------------------
// Acceptance test 3: unsupported column type → InvalidInput with dtype in message
// ---------------------------------------------------------------------------

/// ParquetReader on a List<Int32> file must fail at construction with an
/// InvalidInput error whose message mentions the actual column dtype.
#[test]
fn test_unsupported_column_type_returns_error_with_dtype() {
    let item_field = Arc::new(Field::new("item", DataType::Int32, true));
    let list_field = Field::new("data", DataType::List(item_field), true);
    let schema = Arc::new(Schema::new(vec![list_field]));

    let mut builder = arrow::array::ListBuilder::new(arrow::array::Int32Builder::new());
    builder.values().append_value(42);
    builder.append(true);
    let array = Arc::new(builder.finish()) as ArrayRef;

    let tmp = write_list_parquet(schema, vec![array]);

    let result_f32 = ParquetReader::<f32>::new(tmp.path(), None, NullHandling::FillZero);
    let result_f64 = ParquetReader::<f64>::new(tmp.path(), None, NullHandling::FillZero);

    for result in [result_f32.map(|_| ()), result_f64.map(|_| ())] {
        assert!(result.is_err(), "expected error for Int32 column");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Int32") || msg.contains("int32"),
            "error message should contain the dtype, got: {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// Acceptance test 4: FixedSizeList<f32> read as f32 (zero-copy path)
// ---------------------------------------------------------------------------

/// ParquetReader::<f32> on a FixedSizeList<Float32> file → values come back as
/// Vec<f32> with no precision loss.
#[test]
fn test_fixed_size_list_f32_as_f32() {
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let list_field = Field::new("data", DataType::FixedSizeList(item_field, 3), true);
    let schema = Arc::new(Schema::new(vec![list_field]));

    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 3);
    builder.values().append_slice(&[1.0_f32, 2.0_f32, 3.0_f32]);
    builder.append(true);
    builder.values().append_slice(&[4.0_f32, 5.0_f32, 6.0_f32]);
    builder.append(true);
    let array = Arc::new(builder.finish()) as ArrayRef;

    let tmp = write_list_parquet(schema, vec![array]);

    let mut reader = ParquetReader::<f32>::new(tmp.path(), None, NullHandling::FillZero).unwrap();
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();

    assert_eq!(num_samples, 2);
    assert_eq!(sample_size, 3);
    assert_eq!(data, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ---------------------------------------------------------------------------
// Acceptance test 5: FixedSizeList<f64> cast to f32
// ---------------------------------------------------------------------------

/// ParquetReader::<f32> on a FixedSizeList<Float64> file → Arrow cast applied.
#[test]
fn test_fixed_size_list_f64_cast_to_f32() {
    let item_field = Arc::new(Field::new("item", DataType::Float64, true));
    let list_field = Field::new("data", DataType::FixedSizeList(item_field, 2), true);
    let schema = Arc::new(Schema::new(vec![list_field]));

    let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), 2);
    builder.values().append_slice(&[1.5_f64, -2.5_f64]);
    builder.append(true);
    builder
        .values()
        .append_slice(&[f64::from(f32::MAX) * 2.0, f64::NAN]);
    builder.append(true);
    let array = Arc::new(builder.finish()) as ArrayRef;

    let tmp = write_list_parquet(schema, vec![array]);

    let mut reader = ParquetReader::<f32>::new(tmp.path(), None, NullHandling::FillZero).unwrap();
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();

    assert_eq!(num_samples, 2);
    assert_eq!(sample_size, 2);
    assert_eq!(data[0], 1.5_f32);
    assert_eq!(data[1], -2.5_f32);
    assert!(
        data[2].is_infinite() && data[2] > 0.0,
        "expected +Inf, got {}",
        data[2]
    );
    assert!(data[3].is_nan(), "expected NaN, got {}", data[3]);
}

// ---------------------------------------------------------------------------
// Acceptance test 6: ParquetStreamingReader<f32> on f32 column
// ---------------------------------------------------------------------------

/// ParquetStreamingReader::<f32> on a List<Float32> file → same values as
/// ParquetReader::<f32>.
#[test]
fn test_streaming_reader_list_f32() {
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let list_field = Field::new("data", DataType::List(item_field), true);
    let schema = Arc::new(Schema::new(vec![list_field]));

    let mut builder = ListBuilder::new(Float32Builder::new());
    builder.values().append_slice(&[1.0_f32, 2.5_f32, 3.75_f32]);
    builder.append(true);
    builder.values().append_slice(&[4.0_f32, 5.5_f32, 6.25_f32]);
    builder.append(true);
    let array = Arc::new(builder.finish()) as ArrayRef;

    let tmp = write_list_parquet(schema, vec![array]);

    let mut reader =
        ParquetStreamingReader::<f32>::new(tmp.path(), None, NullHandling::FillZero).unwrap();
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();

    assert_eq!(num_samples, 2);
    assert_eq!(sample_size, 3);
    assert_eq!(data, vec![1.0_f32, 2.5, 3.75, 4.0, 5.5, 6.25]);
}

// ---------------------------------------------------------------------------
// Acceptance test 7: ParquetStreamingReader<f32> on f64 column (cast path)
// ---------------------------------------------------------------------------

/// ParquetStreamingReader::<f32> on a List<Float64> file → Arrow cast applied;
/// overflow → ±Inf, NaN preserved.
#[test]
fn test_streaming_reader_list_f64_cast_to_f32() {
    let item_field = Arc::new(Field::new("item", DataType::Float64, true));
    let list_field = Field::new("data", DataType::List(item_field), true);
    let schema = Arc::new(Schema::new(vec![list_field]));

    let mut builder = ListBuilder::new(Float64Builder::new());
    builder.values().append_slice(&[1.0_f64, -2.0_f64]);
    builder.append(true);
    builder.values().append_value(f64::from(f32::MAX) * 2.0);
    builder.values().append_value(f64::NAN);
    builder.append(true);
    let array = Arc::new(builder.finish()) as ArrayRef;

    let tmp = write_list_parquet(schema, vec![array]);

    let mut reader =
        ParquetStreamingReader::<f32>::new(tmp.path(), None, NullHandling::FillZero).unwrap();
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();

    assert_eq!(num_samples, 2);
    assert_eq!(sample_size, 2);
    assert_eq!(data[0], 1.0_f32);
    assert_eq!(data[1], -2.0_f32);
    assert!(
        data[2].is_infinite() && data[2] > 0.0,
        "expected +Inf, got {}",
        data[2]
    );
    assert!(data[3].is_nan(), "expected NaN, got {}", data[3]);
}
