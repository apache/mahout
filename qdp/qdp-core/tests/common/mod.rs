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

use arrow::array::{
    Array, FixedSizeListArray, Float32Builder, Float64Array, Float64Builder, ListBuilder,
};
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
#[allow(clippy::manual_is_multiple_of)]
pub fn write_fixed_size_list_parquet(path: &str, data: &[f64], sample_size: usize) {
    assert!(sample_size > 0, "sample_size must be > 0");
    assert!(
        data.len() % sample_size == 0,
        "Data length ({}) must be a multiple of sample size ({})",
        data.len(),
        sample_size
    );
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

/// Writes a `List<Float32>` Parquet file; each `sample_size` consecutive values in
/// `data` form one row. Mirrors [`write_list_parquet_f64`] for the f32 column path
/// (issue #1342 Parquet f32 fidelity tests).
#[allow(dead_code)]
#[allow(clippy::manual_is_multiple_of)]
pub fn write_list_parquet_f32(path: &str, data: &[f32], sample_size: usize) {
    assert!(sample_size > 0, "sample_size must be > 0");
    assert!(
        data.len() % sample_size == 0,
        "Data length ({}) must be a multiple of sample size ({})",
        data.len(),
        sample_size
    );
    use std::fs::File;
    use std::sync::Arc;

    let mut builder = ListBuilder::new(Float32Builder::new());
    for row in data.chunks(sample_size) {
        builder.values().append_slice(row);
        builder.append(true);
    }
    let list_array = builder.finish();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        list_array.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list_array) as _]).unwrap();

    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Writes a `List<Float64>` Parquet file; each `sample_size` consecutive values in
/// `data` form one row. Companion to [`write_list_parquet_f32`].
#[allow(dead_code)]
#[allow(clippy::manual_is_multiple_of)]
pub fn write_list_parquet_f64(path: &str, data: &[f64], sample_size: usize) {
    assert!(sample_size > 0, "sample_size must be > 0");
    assert!(
        data.len() % sample_size == 0,
        "Data length ({}) must be a multiple of sample size ({})",
        data.len(),
        sample_size
    );
    use std::fs::File;
    use std::sync::Arc;

    let mut builder = ListBuilder::new(Float64Builder::new());
    for row in data.chunks(sample_size) {
        builder.values().append_slice(row);
        builder.append(true);
    }
    let list_array = builder.finish();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        list_array.data_type().clone(),
        true,
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

/// Returns a QDP engine, or `None` when one cannot be created — including on hosts with no
/// NVIDIA driver at all.
///
/// [`qdp_engine`] is not enough for that case: when `libcuda` cannot be dlopened, `cudarc`
/// panics (`panic_no_lib_found`) instead of returning `Err`, so `.ok()` never yields `None` and
/// the test aborts rather than skipping. Probing inside `catch_unwind` with the default hook
/// suppressed borrows that much from `parquet_f32_fidelity.rs`'s helper for issue #1342.
///
/// It stops short of what that helper does, and the difference matters. A stub build — toolkit
/// absent, `libcuda` present — creates an engine *successfully* and only fails when a kernel is
/// launched, so this probe returns `Some` there. That is sufficient for callers that never launch
/// a kernel, which is why the memory-guard tests can use it: they assert on the guard's decision
/// before any encode runs. A test that does launch one needs `parquet_f32_fidelity.rs`'s version,
/// which probes with a trivial 1-qubit encode; using this one instead reproduces #1342.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn qdp_engine_probed() -> Option<QdpEngine> {
    // The panic hook is process-global, so serialize probes against each other. This narrows the
    // suppression window but does not close it: the harness runs tests on parallel threads, so a
    // genuine panic elsewhere that lands inside this window still loses its message. Same hazard
    // as the `parquet_f32_fidelity.rs` helper this borrows from.
    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let probe = std::panic::catch_unwind(|| QdpEngine::new(0).ok());
    std::panic::set_hook(prev_hook);
    probe.ok().flatten()
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

/// Downloads a complex128 DLPack tensor as interleaved real and imaginary values.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub unsafe fn download_dlpack_complex_f64(dlpack_ptr: *mut DLManagedTensor) -> Vec<f64> {
    assert!(!dlpack_ptr.is_null(), "DLPack pointer should not be null");

    let tensor = unsafe { &(*dlpack_ptr).dl_tensor };
    assert!(tensor.ndim > 0, "DLPack tensor should have dimensions");
    assert!(!tensor.shape.is_null(), "DLPack shape should not be null");
    assert!(!tensor.data.is_null(), "DLPack data should not be null");
    assert_eq!(
        tensor.device.device_type,
        qdp_core::dlpack::DLDeviceType::kDLCUDA
    );
    assert_eq!(tensor.dtype.code, qdp_core::dlpack::DL_COMPLEX);
    assert_eq!(tensor.dtype.bits, 128);
    assert_eq!(tensor.dtype.lanes, 1);

    let shape = unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) };
    let num_elements = shape.iter().try_fold(1_usize, |count, &dim| {
        usize::try_from(dim)
            .ok()
            .and_then(|dim| count.checked_mul(dim))
    });
    let num_elements = num_elements.expect("DLPack shape should fit in usize");
    let byte_count = num_elements
        .checked_mul(std::mem::size_of::<qdp_kernels::CuDoubleComplex>())
        .expect("DLPack byte count should fit in usize");
    let host_len = num_elements
        .checked_mul(2)
        .expect("DLPack complex128 element count should fit in usize");
    let mut host = vec![0.0_f64; host_len];
    let device_ptr = (tensor.data as u64)
        .checked_add(tensor.byte_offset)
        .expect("DLPack byte offset should fit in a device pointer");

    let ret = unsafe {
        cudarc::driver::sys::lib().cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), device_ptr, byte_count)
    };
    assert_eq!(ret, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
    host
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
