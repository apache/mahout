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

//
// Tests for GPU-side f32 L2 norm helper in AmplitudeEncoder.
//

#![cfg(target_os = "linux")]

use approx::assert_relative_eq;
use cudarc::driver::{CudaDevice, DevicePtr};
use qdp_core::gpu::encodings::amplitude::AmplitudeEncoder;

#[test]
fn test_calculate_inv_norm_gpu_f32_basic() {
    println!("Testing AmplitudeEncoder::calculate_inv_norm_gpu_f32 (basic case)...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Input: [3.0, 4.0] -> norm = 5.0, inv_norm = 0.2
    let input: Vec<f32> = vec![3.0, 4.0];
    let expected_norm = (3.0_f32.powi(2) + 4.0_f32.powi(2)).sqrt();
    let expected_inv_norm = 1.0_f32 / expected_norm;

    let input_d = device.htod_sync_copy(input.as_slice()).unwrap();
    let inv = unsafe {
        AmplitudeEncoder::calculate_inv_norm_gpu_f32(
            &device,
            *input_d.device_ptr() as *const f32,
            input.len(),
        )
        .unwrap()
    };

    assert_relative_eq!(inv, expected_inv_norm, epsilon = 1e-6_f32);
}

#[test]
fn test_calculate_inv_norm_gpu_f32_invalid_zero() {
    println!("Testing AmplitudeEncoder::calculate_inv_norm_gpu_f32 with zero vector...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let input: Vec<f32> = vec![0.0, 0.0, 0.0];
    let input_d = device.htod_sync_copy(input.as_slice()).unwrap();

    let result = unsafe {
        AmplitudeEncoder::calculate_inv_norm_gpu_f32(
            &device,
            *input_d.device_ptr() as *const f32,
            input.len(),
        )
    };

    assert!(
        result.is_err(),
        "Expected error for zero-norm f32 input, got {:?}",
        result
    );
}
