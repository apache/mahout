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

// Tests for the GPU-side f32 inverse-norm reduction used by amplitude encoding.

#![cfg(target_os = "linux")]

use cudarc::driver::DevicePtr;
use qdp_core::gpu::kernels::{LaunchCtx, Shape, amplitude};

mod common;

/// Queue the reduction, wait, and return (inverse norms, flag).
fn inv_norms_sync(input: &[f32], shape: Shape) -> Option<(Vec<f32>, i32)> {
    let (device, input_d) = common::copy_f32_to_device(input)?;
    let ctx = LaunchCtx::default_stream(&device);
    let (norms, flag) = unsafe {
        amplitude::inv_norms::<f32>(&ctx, *input_d.device_ptr() as *const f32, shape).unwrap()
    };
    device.synchronize().unwrap();
    let norms = device.dtoh_sync_copy(&norms).unwrap();
    let flag = device.dtoh_sync_copy(&flag).unwrap()[0];
    Some((norms, flag))
}

#[test]
fn test_inv_norms_f32_basic() {
    // Input: [3.0, 4.0] -> norm = 5.0, inv_norm = 0.2
    let expected = 1.0_f32 / 5.0;
    let Some((norms, flag)) = inv_norms_sync(&[3.0, 4.0], Shape::new(1, 2)) else {
        println!("SKIP: No CUDA device");
        return;
    };
    assert_eq!(norms.len(), 1);
    assert!((norms[0] - expected).abs() < 1e-6, "got {}", norms[0]);
    assert_eq!(flag, 0);
}

#[test]
fn test_inv_norms_f32_per_sample() {
    // Two samples: [3, 4] and [0, 2] -> inverse norms 0.2 and 0.5
    let Some((norms, flag)) = inv_norms_sync(&[3.0, 4.0, 0.0, 2.0], Shape::new(2, 2)) else {
        println!("SKIP: No CUDA device");
        return;
    };
    assert!(
        (norms[0] - 0.2).abs() < 1e-6 && (norms[1] - 0.5).abs() < 1e-6,
        "{norms:?}"
    );
    assert_eq!(flag, 0);
}

#[test]
fn test_inv_norms_f32_single_large_sample() {
    // 2^15 elements of 1.0: norm = sqrt(2^15), exercised through the wide
    // single-sample reduction rather than the per-sample batch kernel.
    let n = 1usize << 15;
    let input = vec![1.0_f32; n];
    let Some((norms, flag)) = inv_norms_sync(&input, Shape::new(1, n)) else {
        println!("SKIP: No CUDA device");
        return;
    };
    let expected = 1.0 / (n as f32).sqrt();
    assert!((norms[0] - expected).abs() < 1e-6, "got {}", norms[0]);
    assert_eq!(flag, 0);
}

#[test]
fn test_inv_norms_f32_invalid_zero_raises_flag() {
    let Some((norms, flag)) = inv_norms_sync(&[0.0, 0.0, 0.0], Shape::new(1, 3)) else {
        println!("SKIP: No CUDA device");
        return;
    };
    assert_eq!(norms[0], 0.0, "invalid samples get inv_norm 0");
    assert_eq!(flag, 1, "zero norm must raise the flag");
}
