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

// Tests for encode_from_gpu_ptr (f32, f64, etc.)

#![cfg(target_os = "linux")]

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::safe::CudaStream;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use qdp_core::{MahoutError, Precision, QdpEngine};

fn engine_f32() -> Option<QdpEngine> {
    QdpEngine::new_with_precision(0, Precision::Float32).ok()
}

fn device_and_f32_slice(data: &[f32]) -> Option<(Arc<CudaDevice>, CudaSlice<f32>)> {
    let device = CudaDevice::new(0).ok()?;
    let slice = device.htod_sync_copy(data).ok()?;
    Some((device, slice))
}

fn assert_dlpack_shape_2_4(dlpack_ptr: *mut qdp_core::dlpack::DLManagedTensor) {
    assert!(!dlpack_ptr.is_null());
    unsafe {
        let tensor = &(*dlpack_ptr).dl_tensor;
        assert_eq!(tensor.ndim, 2);
        let shape = std::slice::from_raw_parts(tensor.shape, 2);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 4);
        if let Some(deleter) = (*dlpack_ptr).deleter {
            deleter(dlpack_ptr);
        }
    }
}

#[test]
fn test_encode_from_gpu_ptr_f32() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let len = input_d.len();

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_f32(ptr, len, 2)
            .expect("encode_from_gpu_ptr_f32")
    };
    assert_dlpack_shape_2_4(dlpack_ptr);
}

#[test]
fn test_encode_from_gpu_ptr_f32_with_stream() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let len = input_d.len();
    let stream = std::ptr::null_mut::<c_void>();

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_f32_with_stream(ptr, len, 2, stream)
            .expect("encode_from_gpu_ptr_f32_with_stream")
    };
    assert_dlpack_shape_2_4(dlpack_ptr);
}

#[test]
fn test_encode_from_gpu_ptr_f32_with_stream_non_default_stream() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let stream: CudaStream = device.fork_default_stream().expect("fork_default_stream");
    let stream_ptr = stream.stream as *mut c_void;
    let ptr = *input_d.device_ptr() as *const f32;
    let len = input_d.len();

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_f32_with_stream(ptr, len, 2, stream_ptr)
            .expect("encode_from_gpu_ptr_f32_with_stream (non-default stream)")
    };
    assert_dlpack_shape_2_4(dlpack_ptr);
}

#[test]
fn test_encode_from_gpu_ptr_f32_success_when_f64_precision() {
    let engine = match QdpEngine::new_with_precision(0, Precision::Float64).ok() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let len = input_d.len();

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_f32(ptr, len, 2)
            .expect("encode_from_gpu_ptr_f32 (Float64 engine)")
    };
    assert_dlpack_shape_2_4(dlpack_ptr);
}

#[test]
fn test_encode_from_gpu_ptr_f32_with_stream_success_when_f64_precision() {
    let engine = match QdpEngine::new_with_precision(0, Precision::Float64).ok() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let len = input_d.len();
    let stream = std::ptr::null_mut::<c_void>();

    let dlpack_ptr = unsafe {
        engine
            .encode_from_gpu_ptr_f32_with_stream(ptr, len, 2, stream)
            .expect("encode_from_gpu_ptr_f32_with_stream (Float64 engine)")
    };
    assert_dlpack_shape_2_4(dlpack_ptr);
}

#[test]
fn test_encode_from_gpu_ptr_f32_empty_input() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;

    let result = unsafe { engine.encode_from_gpu_ptr_f32(ptr, 0, 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("empty")),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_from_gpu_ptr_f32_with_stream_empty_input() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let stream = std::ptr::null_mut::<c_void>();

    let result = unsafe { engine.encode_from_gpu_ptr_f32_with_stream(ptr, 0, 2, stream) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(msg.contains("empty")),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_from_gpu_ptr_f32_input_exceeds_state_len() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let len = input_d.len();

    let result = unsafe { engine.encode_from_gpu_ptr_f32(ptr, len, 2) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(
            msg.contains("exceeds") || msg.contains("state vector"),
            "expected 'exceeds' or 'state vector', got: {}",
            msg
        ),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}

#[test]
fn test_encode_from_gpu_ptr_f32_with_stream_input_exceeds_state_len() {
    let engine = match engine_f32() {
        Some(e) => e,
        None => {
            println!("SKIP: No GPU");
            return;
        }
    };
    let (_device, input_d) = match device_and_f32_slice(&[1.0, 0.0, 0.0, 0.0, 0.0]) {
        Some(t) => t,
        None => {
            println!("SKIP: No CUDA device");
            return;
        }
    };
    let ptr = *input_d.device_ptr() as *const f32;
    let len = input_d.len();
    let stream = std::ptr::null_mut::<c_void>();

    let result = unsafe { engine.encode_from_gpu_ptr_f32_with_stream(ptr, len, 2, stream) };
    assert!(result.is_err());
    match &result.unwrap_err() {
        MahoutError::InvalidInput(msg) => assert!(
            msg.contains("exceeds") || msg.contains("state vector"),
            "expected 'exceeds' or 'state vector', got: {}",
            msg
        ),
        e => panic!("Expected InvalidInput, got {:?}", e),
    }
}
