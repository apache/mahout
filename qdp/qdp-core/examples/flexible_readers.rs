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

//! Example demonstrating the flexible reader architecture
//!
//! Run: cargo run -p qdp-core --example flexible_readers

use qdp_core::reader::DataReader;
use qdp_core::readers::{ArrowIPCReader, NumpyReader};
use arrow::array::{Float64Array, FixedSizeListArray, ListBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter as ArrowFileWriter;
use ndarray::Array2;
use std::fs::File;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QDP Flexible Reader Architecture Demo ===\n");

    // Create some sample quantum state vectors
    let num_samples = 5;
    let sample_size = 8; // 2^3 for 3 qubits
    let mut all_data = Vec::with_capacity(num_samples * sample_size);

    println!("Generating {} samples of size {}...", num_samples, sample_size);
    for i in 0..num_samples {
        for j in 0..sample_size {
            all_data.push((i * sample_size + j) as f64 / (num_samples * sample_size) as f64);
        }
    }

    // === Example 1: Arrow IPC with FixedSizeList ===
    println!("\n[Example 1] Writing and reading Arrow IPC (FixedSizeList format)...");

    let arrow_fixed_path = "/tmp/quantum_states_fixed.arrow";

    // Write Arrow IPC file with FixedSizeList
    let values_array = Float64Array::from(all_data.clone());
    let field = Arc::new(Field::new("item", DataType::Float64, false));
    let list_array = FixedSizeListArray::new(
        field,
        sample_size as i32,
        Arc::new(values_array),
        None,
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "quantum_state",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float64, false)),
            sample_size as i32,
        ),
        false,
    )]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(list_array)],
    )?;

    let file = File::create(arrow_fixed_path)?;
    let mut writer = ArrowFileWriter::try_new(file, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    println!("  Written to: {}", arrow_fixed_path);

    let mut arrow_reader = ArrowIPCReader::new(arrow_fixed_path)?;
    let (data, samples, size) = arrow_reader.read_batch()?;
    println!("  Read {} samples of size {}", samples, size);
    println!("  First sample: {:?}", &data[0..size]);

    // === Example 2: Arrow IPC with List (variable length) ===
    println!("\n[Example 2] Writing and reading Arrow IPC (List format)...");

    let arrow_list_path = "/tmp/quantum_states_list.arrow";

    // Write Arrow IPC file with List
    let mut list_builder = ListBuilder::new(Float64Array::builder(num_samples * sample_size));

    for i in 0..num_samples {
        let values: Vec<f64> = (0..sample_size)
            .map(|j| (i * sample_size + j) as f64 / (num_samples * sample_size) as f64)
            .collect();
        list_builder.values().append_slice(&values);
        list_builder.append(true);
    }

    let list_array = list_builder.finish();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "quantum_state",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        false,
    )]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(list_array)],
    )?;

    let file = File::create(arrow_list_path)?;
    let mut writer = ArrowFileWriter::try_new(file, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    println!("  Written to: {}", arrow_list_path);

    let mut arrow_reader = ArrowIPCReader::new(arrow_list_path)?;
    let (data, samples, size) = arrow_reader.read_batch()?;
    println!("  Read {} samples of size {}", samples, size);
    println!("  First sample: {:?}", &data[0..size]);

    // === Example 3: NumPy Format ===
    println!("\n[Example 3] Writing and reading NumPy format...");

    let numpy_path = "/tmp/quantum_states.npy";

    // Create and write NumPy array
    let array = Array2::from_shape_vec((num_samples, sample_size), all_data.clone())?;
    ndarray_npy::write_npy(numpy_path, &array)?;
    println!("  Written to: {}", numpy_path);

    let mut numpy_reader = NumpyReader::new(numpy_path)?;
    let (data, samples, size) = numpy_reader.read_batch()?;
    println!("  Read {} samples of size {}", samples, size);
    println!("  First sample: {:?}", &data[0..size]);

    // === Example 4: Demonstrating Generic Reader Usage ===
    println!("\n[Example 4] Using readers polymorphically...");

    fn process_with_any_reader<R: DataReader>(mut reader: R, format_name: &str)
        -> Result<(), Box<dyn std::error::Error>>
    {
        let (data, samples, size) = reader.read_batch()?;
        println!("  {} format: {} samples × {} elements = {} total",
                 format_name, samples, size, data.len());
        Ok(())
    }

    let arrow_reader = ArrowIPCReader::new(arrow_fixed_path)?;
    process_with_any_reader(arrow_reader, "Arrow IPC (FixedSizeList)")?;

    let arrow_reader = ArrowIPCReader::new(arrow_list_path)?;
    process_with_any_reader(arrow_reader, "Arrow IPC (List)")?;

    let numpy_reader = NumpyReader::new(numpy_path)?;
    process_with_any_reader(numpy_reader, "NumPy")?;

    // === Example 5: Format Detection Pattern ===
    println!("\n[Example 5] Automatic format detection pattern...");

    fn read_any_format(path: &str) -> Result<(Vec<f64>, usize, usize), Box<dyn std::error::Error>> {
        if path.ends_with(".parquet") {
            Err("Parquet needs List<Float64> format - use write_parquet_batch helper".into())
        } else if path.ends_with(".arrow") || path.ends_with(".feather") {
            let mut reader = ArrowIPCReader::new(path)?;
            Ok(reader.read_batch()?)
        } else if path.ends_with(".npy") {
            let mut reader = NumpyReader::new(path)?;
            Ok(reader.read_batch()?)
        } else {
            Err("Unsupported format".into())
        }
    }

    let (_, samples, size) = read_any_format(arrow_fixed_path)?;
    println!("  Auto-detected Arrow (FixedSizeList): {} samples × {}", samples, size);

    let (_, samples, size) = read_any_format(arrow_list_path)?;
    println!("  Auto-detected Arrow (List): {} samples × {}", samples, size);

    let (_, samples, size) = read_any_format(numpy_path)?;
    println!("  Auto-detected NumPy: {} samples × {}", samples, size);

    println!("\n=== Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Different formats implement the same DataReader trait");
    println!("  2. Readers can be used polymorphically via trait objects");
    println!("  3. Easy to add new formats without changing existing code");
    println!("  4. Format detection can be handled at a higher level");
    println!("  5. Arrow FixedSizeList, List, and NumPy formats are supported");
    println!("\nSee docs/ADDING_INPUT_FORMATS.md for how to add new formats!");

    Ok(())
}
