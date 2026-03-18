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

#[cfg(target_os = "linux")]
use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaDevice, CudaSlice};
#[cfg(target_os = "linux")]
use qdp_core::dlpack::DLManagedTensor;
#[cfg(target_os = "linux")]
use qdp_core::{Precision, QdpEngine};

/// Creates normalized test data (f64)
#[allow(dead_code)] // Used by multiple test modules
pub fn create_test_data(size: usize) -> Vec<f64> {
    (0..size).map(|i| (i as f64) / (size as f64)).collect()
}

/// Creates normalized test data (f32)
#[allow(dead_code)]
pub fn create_test_data_f32(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) / (size as f32)).collect()
}

/// Writes a FixedSizeList<Float64, sample_size> Parquet file for streaming encoder tests.
/// Each `sample_size` consecutive values in `data` form one row.
#[allow(dead_code)]
pub fn write_fixed_size_list_parquet(path: &str, data: &[f64], sample_size: usize) {
    use std::fs::File;
    use std::sync::Arc;

    let item_field = Arc::new(Field::new("item", DataType::Float64, false));
    let values_array = Float64Array::from(data.to_vec());
    let list_array = FixedSizeListArray::new(
        item_field.clone(),
        sample_size as i32,
        Arc::new(values_array),
        None,
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "angles",
        DataType::FixedSizeList(item_field, sample_size as i32),
        false,
    )]));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list_array) as _]).unwrap();

    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Returns a CUDA device handle, or `None` when CUDA is unavailable for the test environment.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn cuda_device() -> Option<Arc<CudaDevice>> {
    CudaDevice::new(0).ok()
}

/// Returns a QDP engine, or `None` when GPU-backed engine initialization is unavailable.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn qdp_engine() -> Option<QdpEngine> {
    QdpEngine::new(0).ok()
}

/// Returns a QDP engine with the requested precision, or `None` when unavailable.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn qdp_engine_with_precision(precision: Precision) -> Option<QdpEngine> {
    QdpEngine::new_with_precision(0, precision).ok()
}

/// Copies f64 host data to the default CUDA device, or returns `None` when unavailable.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn copy_f64_to_device(data: &[f64]) -> Option<(Arc<CudaDevice>, CudaSlice<f64>)> {
    let device = cuda_device()?;
    let slice = device.htod_sync_copy(data).ok()?;
    Some((device, slice))
}

/// Copies f32 host data to the default CUDA device, or returns `None` when unavailable.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn copy_f32_to_device(data: &[f32]) -> Option<(Arc<CudaDevice>, CudaSlice<f32>)> {
    let device = cuda_device()?;
    let slice = device.htod_sync_copy(data).ok()?;
    Some((device, slice))
}

/// Copies usize host data to the default CUDA device, or returns `None` when unavailable.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn copy_usize_to_device(data: &[usize]) -> Option<(Arc<CudaDevice>, CudaSlice<usize>)> {
    let device = cuda_device()?;
    let slice = device.htod_sync_copy(data).ok()?;
    Some((device, slice))
}

/// Asserts a DLPack tensor is 2D with the expected shape.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub unsafe fn assert_dlpack_shape_2d(dlpack_ptr: *mut DLManagedTensor, dim0: i64, dim1: i64) {
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    let tensor = unsafe { &(*dlpack_ptr).dl_tensor };
    assert_eq!(tensor.ndim, 2, "DLPack tensor should be 2D");

    let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 2) };
    assert_eq!(shape[0], dim0, "Unexpected first dimension");
    assert_eq!(shape[1], dim1, "Unexpected second dimension");
}

/// Asserts a DLPack tensor is 2D with the expected shape and then frees it via its deleter.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub unsafe fn assert_dlpack_shape_2d_and_delete(
    dlpack_ptr: *mut DLManagedTensor,
    dim0: i64,
    dim1: i64,
) {
    unsafe { assert_dlpack_shape_2d(dlpack_ptr, dim0, dim1) };

    unsafe { take_deleter_and_delete(dlpack_ptr) };
}

/// Takes the DLPack deleter from the managed tensor and invokes it exactly once.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub unsafe fn take_deleter_and_delete(dlpack_ptr: *mut DLManagedTensor) {
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    let managed = unsafe { &mut *dlpack_ptr };
    let deleter = managed
        .deleter
        .take()
        .expect("DLPack deleter should be present");
    unsafe { deleter(dlpack_ptr) };
}
