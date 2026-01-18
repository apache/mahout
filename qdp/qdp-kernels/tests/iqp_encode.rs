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

// Tests for IQP encoding CUDA kernel

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
#[cfg(target_os = "linux")]
use qdp_kernels::{CuDoubleComplex, launch_iqp_encode, launch_iqp_encode_batch};

const EPSILON: f64 = 1e-10;

/// Entanglement types (must match CUDA kernel enum)
const IQP_NONE: i32 = 0;
const IQP_LINEAR: i32 = 1;
const IQP_FULL: i32 = 2;

/// CPU reference implementation for IQP encoding (no entanglement)
#[cfg(target_os = "linux")]
fn iqp_encode_reference_none(input: &[f64], num_qubits: usize) -> Vec<(f64, f64)> {
    let state_len = 1 << num_qubits;
    let norm_factor = 1.0 / (state_len as f64).sqrt();

    (0..state_len)
        .map(|z| {
            // Phase = Σ_i x_i * z_i (where z_i is bit i of z)
            let mut phase = 0.0;
            for (i, &x) in input.iter().enumerate() {
                if i >= num_qubits {
                    break;
                }
                if (z >> i) & 1 == 1 {
                    phase += x;
                }
            }
            let (sin_phase, cos_phase) = phase.sin_cos();
            (norm_factor * cos_phase, norm_factor * sin_phase)
        })
        .collect()
}

/// CPU reference implementation for IQP encoding (linear entanglement)
#[cfg(target_os = "linux")]
fn iqp_encode_reference_linear(input: &[f64], num_qubits: usize) -> Vec<(f64, f64)> {
    let state_len = 1 << num_qubits;
    let norm_factor = 1.0 / (state_len as f64).sqrt();
    let effective_n = input.len().min(num_qubits);

    (0..state_len)
        .map(|z| {
            let mut phase = 0.0;

            // Single-qubit terms
            for (i, &x) in input.iter().enumerate().take(effective_n) {
                if (z >> i) & 1 == 1 {
                    phase += x;
                }
            }

            // Linear entanglement terms
            for i in 0..(effective_n.saturating_sub(1)) {
                let bit_i = (z >> i) & 1;
                let bit_i1 = (z >> (i + 1)) & 1;
                if bit_i == 1 && bit_i1 == 1 {
                    phase += input[i] * input[i + 1];
                }
            }

            let (sin_phase, cos_phase) = phase.sin_cos();
            (norm_factor * cos_phase, norm_factor * sin_phase)
        })
        .collect()
}

/// CPU reference implementation for IQP encoding (full entanglement)
#[cfg(target_os = "linux")]
fn iqp_encode_reference_full(input: &[f64], num_qubits: usize) -> Vec<(f64, f64)> {
    let state_len = 1 << num_qubits;
    let norm_factor = 1.0 / (state_len as f64).sqrt();
    let effective_n = input.len().min(num_qubits);

    (0..state_len)
        .map(|z| {
            let mut phase = 0.0;

            // Single-qubit terms
            for (i, &x) in input.iter().enumerate().take(effective_n) {
                if (z >> i) & 1 == 1 {
                    phase += x;
                }
            }

            // Full entanglement terms
            for i in 0..effective_n {
                let bit_i = (z >> i) & 1;
                if bit_i == 0 {
                    continue;
                }
                for j in (i + 1)..effective_n {
                    let bit_j = (z >> j) & 1;
                    if bit_j == 1 {
                        phase += input[i] * input[j];
                    }
                }
            }

            let (sin_phase, cos_phase) = phase.sin_cos();
            (norm_factor * cos_phase, norm_factor * sin_phase)
        })
        .collect()
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_basic_no_entanglement() {
    println!("Testing basic IQP encoding (no entanglement)...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Test input: 2 features, 2 qubits
    let input = vec![0.5, 1.0];
    let num_qubits = 2;
    let state_len = 4;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_NONE,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    let reference = iqp_encode_reference_none(&input, num_qubits);

    for (i, (gpu, cpu)) in state_h.iter().zip(reference.iter()).enumerate() {
        assert!(
            (gpu.x - cpu.0).abs() < EPSILON,
            "Element {} real: GPU={}, CPU={}",
            i,
            gpu.x,
            cpu.0
        );
        assert!(
            (gpu.y - cpu.1).abs() < EPSILON,
            "Element {} imag: GPU={}, CPU={}",
            i,
            gpu.y,
            cpu.1
        );
    }

    // Verify normalization: sum of |amplitude|^2 should be 1
    let total_prob: f64 = state_h.iter().map(|c| c.x * c.x + c.y * c.y).sum();
    assert!(
        (total_prob - 1.0).abs() < EPSILON,
        "Total probability should be 1.0, got {}",
        total_prob
    );

    println!("PASS: Basic IQP encoding (no entanglement) works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_linear_entanglement() {
    println!("Testing IQP encoding with linear entanglement...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Test input: 3 features, 3 qubits
    let input = vec![0.3, 0.5, 0.7];
    let num_qubits = 3;
    let state_len = 8;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_LINEAR,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    let reference = iqp_encode_reference_linear(&input, num_qubits);

    for (i, (gpu, cpu)) in state_h.iter().zip(reference.iter()).enumerate() {
        assert!(
            (gpu.x - cpu.0).abs() < EPSILON,
            "Element {} real: GPU={}, CPU={}",
            i,
            gpu.x,
            cpu.0
        );
        assert!(
            (gpu.y - cpu.1).abs() < EPSILON,
            "Element {} imag: GPU={}, CPU={}",
            i,
            gpu.y,
            cpu.1
        );
    }

    let total_prob: f64 = state_h.iter().map(|c| c.x * c.x + c.y * c.y).sum();
    assert!(
        (total_prob - 1.0).abs() < EPSILON,
        "Total probability should be 1.0, got {}",
        total_prob
    );

    println!("PASS: IQP encoding with linear entanglement works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_full_entanglement() {
    println!("Testing IQP encoding with full entanglement...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Test input: 3 features, 3 qubits
    let input = vec![0.2, 0.4, 0.6];
    let num_qubits = 3;
    let state_len = 8;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_FULL,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    let reference = iqp_encode_reference_full(&input, num_qubits);

    for (i, (gpu, cpu)) in state_h.iter().zip(reference.iter()).enumerate() {
        assert!(
            (gpu.x - cpu.0).abs() < EPSILON,
            "Element {} real: GPU={}, CPU={}",
            i,
            gpu.x,
            cpu.0
        );
        assert!(
            (gpu.y - cpu.1).abs() < EPSILON,
            "Element {} imag: GPU={}, CPU={}",
            i,
            gpu.y,
            cpu.1
        );
    }

    let total_prob: f64 = state_h.iter().map(|c| c.x * c.x + c.y * c.y).sum();
    assert!(
        (total_prob - 1.0).abs() < EPSILON,
        "Total probability should be 1.0, got {}",
        total_prob
    );

    println!("PASS: IQP encoding with full entanglement works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_fewer_features_than_qubits() {
    println!("Testing IQP encoding with fewer features than qubits...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // 2 features, 4 qubits - extra qubits should have no phase contribution
    let input = vec![1.0, 2.0];
    let num_qubits = 4;
    let state_len = 16;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_NONE,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    let reference = iqp_encode_reference_none(&input, num_qubits);

    for (i, (gpu, cpu)) in state_h.iter().zip(reference.iter()).enumerate() {
        assert!(
            (gpu.x - cpu.0).abs() < EPSILON,
            "Element {} real mismatch",
            i
        );
        assert!(
            (gpu.y - cpu.1).abs() < EPSILON,
            "Element {} imag mismatch",
            i
        );
    }

    let total_prob: f64 = state_h.iter().map(|c| c.x * c.x + c.y * c.y).sum();
    assert!(
        (total_prob - 1.0).abs() < EPSILON,
        "Total probability should be 1.0"
    );

    println!("PASS: IQP encoding with fewer features than qubits works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_more_features_than_qubits() {
    println!("Testing IQP encoding with more features than qubits...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // 5 features, 3 qubits - only first 3 features should be used
    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let num_qubits = 3;
    let state_len = 8;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_FULL,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    let reference = iqp_encode_reference_full(&input, num_qubits);

    for (i, (gpu, cpu)) in state_h.iter().zip(reference.iter()).enumerate() {
        assert!(
            (gpu.x - cpu.0).abs() < EPSILON,
            "Element {} real mismatch",
            i
        );
        assert!(
            (gpu.y - cpu.1).abs() < EPSILON,
            "Element {} imag mismatch",
            i
        );
    }

    let total_prob: f64 = state_h.iter().map(|c| c.x * c.x + c.y * c.y).sum();
    assert!(
        (total_prob - 1.0).abs() < EPSILON,
        "Total probability should be 1.0"
    );

    println!("PASS: IQP encoding with more features than qubits works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_large_state() {
    println!("Testing IQP encoding with large state vector (10 qubits)...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let num_qubits = 10;
    let num_features = 10;
    let state_len = 1 << num_qubits; // 1024

    let input: Vec<f64> = (0..num_features).map(|i| (i as f64) * 0.1).collect();

    let input_d = device.htod_copy(input.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_LINEAR,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();
    let reference = iqp_encode_reference_linear(&input, num_qubits);

    // Spot check several elements
    for i in [0, 100, 500, 1023] {
        assert!(
            (state_h[i].x - reference[i].0).abs() < EPSILON,
            "Element {} real mismatch",
            i
        );
        assert!(
            (state_h[i].y - reference[i].1).abs() < EPSILON,
            "Element {} imag mismatch",
            i
        );
    }

    let total_prob: f64 = state_h.iter().map(|c| c.x * c.x + c.y * c.y).sum();
    assert!(
        (total_prob - 1.0).abs() < EPSILON,
        "Total probability should be 1.0"
    );

    println!("PASS: IQP encoding with large state vector works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_zero_input() {
    println!("Testing IQP encoding with zero input...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // All zeros should produce uniform superposition (all phases = 0)
    let input = vec![0.0, 0.0, 0.0];
    let num_qubits = 3;
    let state_len = 8;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_NONE,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    // Expected: all amplitudes = 1/√8 with zero phase
    let expected_amplitude = 1.0 / (state_len as f64).sqrt();

    for (i, c) in state_h.iter().enumerate() {
        assert!(
            (c.x - expected_amplitude).abs() < EPSILON,
            "Element {} real should be {}",
            i,
            expected_amplitude
        );
        assert!(
            c.y.abs() < EPSILON,
            "Element {} imag should be 0 (uniform superposition)",
            i
        );
    }

    println!("PASS: IQP encoding with zero input produces uniform superposition");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_single_qubit() {
    println!("Testing IQP encoding with single qubit...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let input = vec![std::f64::consts::PI / 2.0];
    let num_qubits = 1;
    let state_len = 2;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_NONE,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    let norm = 1.0 / std::f64::consts::SQRT_2;

    // |0⟩: phase = 0, so amplitude = 1/√2 * (1 + 0i)
    assert!(
        (state_h[0].x - norm).abs() < EPSILON,
        "Element 0 real should be 1/√2"
    );
    assert!(state_h[0].y.abs() < EPSILON, "Element 0 imag should be 0");

    // |1⟩: phase = π/2, so amplitude = 1/√2 * (cos(π/2) + i*sin(π/2)) = 1/√2 * (0 + i)
    assert!(
        state_h[1].x.abs() < EPSILON,
        "Element 1 real should be ~0 (cos(π/2))"
    );
    assert!(
        (state_h[1].y - norm).abs() < EPSILON,
        "Element 1 imag should be 1/√2 (sin(π/2))"
    );

    println!("PASS: IQP encoding with single qubit works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_batch_basic() {
    println!("Testing batch IQP encoding...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let num_samples = 3;
    let num_features = 2;
    let num_qubits = 2;
    let state_len = 4;

    // Batch data: [sample0_features, sample1_features, sample2_features]
    let batch_input: Vec<f64> = vec![
        0.1, 0.2, // Sample 0
        0.5, 0.6, // Sample 1
        1.0, 1.5, // Sample 2
    ];

    let input_d = device.htod_copy(batch_input.clone()).unwrap();
    let mut state_d = device
        .alloc_zeros::<CuDoubleComplex>(num_samples * state_len)
        .unwrap();

    let result = unsafe {
        launch_iqp_encode_batch(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            num_samples,
            num_features,
            num_qubits,
            state_len,
            IQP_NONE,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Batch kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    // Verify each sample against single-sample reference
    for sample_idx in 0..num_samples {
        let sample_data: Vec<f64> =
            batch_input[sample_idx * num_features..(sample_idx + 1) * num_features].to_vec();
        let reference = iqp_encode_reference_none(&sample_data, num_qubits);

        for (i, cpu) in reference.iter().enumerate() {
            let gpu_idx = sample_idx * state_len + i;
            assert!(
                (state_h[gpu_idx].x - cpu.0).abs() < EPSILON,
                "Sample {} element {} real mismatch",
                sample_idx,
                i
            );
            assert!(
                (state_h[gpu_idx].y - cpu.1).abs() < EPSILON,
                "Sample {} element {} imag mismatch",
                sample_idx,
                i
            );
        }

        // Verify normalization per sample
        let sample_prob: f64 = state_h[sample_idx * state_len..(sample_idx + 1) * state_len]
            .iter()
            .map(|c| c.x * c.x + c.y * c.y)
            .sum();
        assert!(
            (sample_prob - 1.0).abs() < EPSILON,
            "Sample {} should be normalized",
            sample_idx
        );
    }

    println!("PASS: Batch IQP encoding works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_batch_with_entanglement() {
    println!("Testing batch IQP encoding with full entanglement...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let num_samples = 2;
    let num_features = 3;
    let num_qubits = 3;
    let state_len = 8;

    let batch_input: Vec<f64> = vec![
        0.2, 0.3, 0.4, // Sample 0
        0.5, 0.6, 0.7, // Sample 1
    ];

    let input_d = device.htod_copy(batch_input.clone()).unwrap();
    let mut state_d = device
        .alloc_zeros::<CuDoubleComplex>(num_samples * state_len)
        .unwrap();

    let result = unsafe {
        launch_iqp_encode_batch(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            num_samples,
            num_features,
            num_qubits,
            state_len,
            IQP_FULL,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Batch kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    // Verify each sample
    for sample_idx in 0..num_samples {
        let sample_data: Vec<f64> =
            batch_input[sample_idx * num_features..(sample_idx + 1) * num_features].to_vec();
        let reference = iqp_encode_reference_full(&sample_data, num_qubits);

        for (i, cpu) in reference.iter().enumerate() {
            let gpu_idx = sample_idx * state_len + i;
            assert!(
                (state_h[gpu_idx].x - cpu.0).abs() < EPSILON,
                "Sample {} element {} real mismatch: GPU={}, CPU={}",
                sample_idx,
                i,
                state_h[gpu_idx].x,
                cpu.0
            );
            assert!(
                (state_h[gpu_idx].y - cpu.1).abs() < EPSILON,
                "Sample {} element {} imag mismatch: GPU={}, CPU={}",
                sample_idx,
                i,
                state_h[gpu_idx].y,
                cpu.1
            );
        }
    }

    println!("PASS: Batch IQP encoding with full entanglement works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_invalid_inputs() {
    println!("Testing IQP encoding error handling...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let input = vec![0.1, 0.2];
    let input_d = device.htod_copy(input.clone()).unwrap();
    let mut state_d = device.alloc_zeros::<CuDoubleComplex>(4).unwrap();

    // Test: state_len = 0
    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            2,
            0, // Invalid
            IQP_NONE,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Should reject state_len = 0");

    // Test: num_qubits = 0
    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            0, // Invalid
            4,
            IQP_NONE,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Should reject num_qubits = 0");

    // Test: state_len != 2^num_qubits
    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            2,
            5, // Invalid: should be 4
            IQP_NONE,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Should reject mismatched state_len");

    // Test: num_features = 0
    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            0, // Invalid
            2,
            4,
            IQP_NONE,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Should reject num_features = 0");

    // Test: invalid entanglement type
    let result = unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_d.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            2,
            4,
            99, // Invalid entanglement type
            std::ptr::null_mut(),
        )
    };
    assert_ne!(result, 0, "Should reject invalid entanglement type");

    println!("PASS: IQP encoding error handling works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_iqp_encode_phase_consistency() {
    println!("Testing IQP encoding phase consistency across entanglement types...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Use single feature - all entanglement types should produce same result
    let input = vec![0.5];
    let num_qubits = 2;
    let state_len = 4;

    let input_d = device.htod_copy(input.clone()).unwrap();

    let mut state_none = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();
    let mut state_linear = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();
    let mut state_full = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    // Launch all three variants
    unsafe {
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_none.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_NONE,
            std::ptr::null_mut(),
        );
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_linear.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_LINEAR,
            std::ptr::null_mut(),
        );
        launch_iqp_encode(
            *input_d.device_ptr() as *const f64,
            *state_full.device_ptr_mut() as *mut std::ffi::c_void,
            input.len(),
            num_qubits,
            state_len,
            IQP_FULL,
            std::ptr::null_mut(),
        );
    }

    let h_none = device.dtoh_sync_copy(&state_none).unwrap();
    let h_linear = device.dtoh_sync_copy(&state_linear).unwrap();
    let h_full = device.dtoh_sync_copy(&state_full).unwrap();

    // With single feature, all entanglement types should be identical
    // (no pairs to entangle)
    for i in 0..state_len {
        assert!(
            (h_none[i].x - h_linear[i].x).abs() < EPSILON,
            "Element {} none vs linear real mismatch",
            i
        );
        assert!(
            (h_none[i].y - h_linear[i].y).abs() < EPSILON,
            "Element {} none vs linear imag mismatch",
            i
        );
        assert!(
            (h_none[i].x - h_full[i].x).abs() < EPSILON,
            "Element {} none vs full real mismatch",
            i
        );
        assert!(
            (h_none[i].y - h_full[i].y).abs() < EPSILON,
            "Element {} none vs full imag mismatch",
            i
        );
    }

    println!("PASS: IQP encoding phase consistency verified");
}

#[test]
#[cfg(not(target_os = "linux"))]
fn test_iqp_encode_dummy_non_linux() {
    println!("Testing dummy implementation on non-Linux platform...");

    let result = unsafe {
        qdp_kernels::launch_iqp_encode(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 999, "Dummy implementation should return 999");
    println!("PASS: Non-Linux dummy implementation returns expected error code");
}
