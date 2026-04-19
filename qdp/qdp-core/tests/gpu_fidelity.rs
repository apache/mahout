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

//! Tests for fidelity / trace-distance metrics and F32 vs F64 precision
//! comparison across different qubit counts.

#![cfg(target_os = "linux")]

use approx::assert_relative_eq;
use qdp_core::gpu::metrics::{
    fidelity_cross_precision, fidelity_f32, fidelity_f64, trace_distance_f64,
};

mod common;

// ═══════════════════════════════════════════════════════════════════════
// Unit tests for the fidelity / trace-distance functions themselves
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_fidelity_identical_states() {
    // |0⟩ = (1+0i, 0+0i)  interleaved: [1, 0, 0, 0]
    let state = vec![1.0, 0.0, 0.0, 0.0];
    let f = fidelity_f64(&state, &state).unwrap();
    assert_relative_eq!(f, 1.0, epsilon = 1e-12);
}

#[test]
fn test_fidelity_orthogonal_states() {
    // |0⟩ and |1⟩
    let state_0 = vec![1.0, 0.0, 0.0, 0.0];
    let state_1 = vec![0.0, 0.0, 1.0, 0.0];
    let f = fidelity_f64(&state_0, &state_1).unwrap();
    assert_relative_eq!(f, 0.0, epsilon = 1e-12);
}

#[test]
fn test_fidelity_superposition() {
    // |+⟩ = 1/√2 (|0⟩ + |1⟩)
    let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
    let plus = vec![inv_sqrt2, 0.0, inv_sqrt2, 0.0];
    let f = fidelity_f64(&plus, &plus).unwrap();
    assert_relative_eq!(f, 1.0, epsilon = 1e-12);

    // ⟨+|0⟩ = 1/√2, fidelity = 0.5
    let zero = vec![1.0, 0.0, 0.0, 0.0];
    let f2 = fidelity_f64(&plus, &zero).unwrap();
    assert_relative_eq!(f2, 0.5, epsilon = 1e-12);
}

#[test]
fn test_fidelity_global_phase() {
    // |ψ⟩ = (0+i) * |0⟩ = [0, 1, 0, 0]  (global phase e^{iπ/2})
    let state_a = vec![1.0, 0.0, 0.0, 0.0];
    let state_b = vec![0.0, 1.0, 0.0, 0.0];
    let f = fidelity_f64(&state_a, &state_b).unwrap();
    // |⟨ψ|φ⟩|² = |⟨0| (i|0⟩)|² = |i|² = 1
    assert_relative_eq!(f, 1.0, epsilon = 1e-12);
}

#[test]
fn test_trace_distance_identical() {
    let state = vec![1.0, 0.0, 0.0, 0.0];
    let td = trace_distance_f64(&state, &state).unwrap();
    assert_relative_eq!(td, 0.0, epsilon = 1e-12);
}

#[test]
fn test_trace_distance_orthogonal() {
    let state_0 = vec![1.0, 0.0, 0.0, 0.0];
    let state_1 = vec![0.0, 0.0, 1.0, 0.0];
    let td = trace_distance_f64(&state_0, &state_1).unwrap();
    assert_relative_eq!(td, 1.0, epsilon = 1e-12);
}

#[test]
fn test_fidelity_f32_basic() {
    let state: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let f = fidelity_f32(&state, &state).unwrap();
    assert_relative_eq!(f, 1.0, epsilon = 1e-6);
}

#[test]
fn test_fidelity_cross_precision_basic() {
    let state_f32: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let state_f64: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0];
    let f = fidelity_cross_precision(&state_f32, &state_f64).unwrap();
    assert_relative_eq!(f, 1.0, epsilon = 1e-6);
}

#[test]
fn test_fidelity_length_mismatch_errors() {
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0, 0.0];
    assert!(fidelity_f64(&a, &b).is_err());
}

#[test]
fn test_fidelity_odd_length_errors() {
    let a = vec![1.0, 0.0, 0.5];
    assert!(fidelity_f64(&a, &a).is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// GPU-backed F32 vs F64 precision comparison tests
// ═══════════════════════════════════════════════════════════════════════

use qdp_core::Precision;
use qdp_core::gpu::metrics::{download_complex_f32, download_complex_f64};

/// Encode the same data at F32 and F64, download both, compute cross-precision fidelity.
fn compare_f32_f64_amplitude(num_qubits: usize) -> Option<f64> {
    let engine_f64 = common::qdp_engine_with_precision(Precision::Float64)?;
    let engine_f32 = common::qdp_engine_with_precision(Precision::Float32)?;
    let device = common::cuda_device()?;

    let state_dim = 1usize << num_qubits;
    let num_samples = 4;
    let sample_size = state_dim;

    // Build deterministic test data
    let data_f64: Vec<f64> = (0..num_samples * sample_size)
        .map(|i| ((i as f64) + 1.0) / (num_samples * sample_size) as f64)
        .collect();
    let data_f32: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();

    // Encode at F64
    let dlpack_f64 = engine_f64
        .encode_batch(&data_f64, num_samples, sample_size, num_qubits, "amplitude")
        .expect("F64 encode_batch should succeed");

    // Encode at F32
    let dlpack_f32 = engine_f32
        .encode_batch_f32(&data_f32, num_samples, sample_size, num_qubits, "amplitude")
        .expect("F32 encode_batch should succeed");

    // Read back from GPU
    let f64_tensor = unsafe { &(*dlpack_f64).dl_tensor };
    let f32_tensor = unsafe { &(*dlpack_f32).dl_tensor };

    let total_elements = num_samples * state_dim;
    let host_f64 =
        download_complex_f64(&device, f64_tensor.data as *const _, total_elements).unwrap();
    let host_f32 =
        download_complex_f32(&device, f32_tensor.data as *const _, total_elements).unwrap();

    // Compute per-sample fidelity, return minimum
    let mut min_fidelity = 1.0_f64;
    for s in 0..num_samples {
        let offset_f64 = s * state_dim * 2;
        let offset_f32 = s * state_dim * 2;
        let sample_f64 = &host_f64[offset_f64..offset_f64 + state_dim * 2];
        let sample_f32 = &host_f32[offset_f32..offset_f32 + state_dim * 2];
        let f = fidelity_cross_precision(sample_f32, sample_f64).unwrap();
        if f < min_fidelity {
            min_fidelity = f;
        }
    }

    // Clean up DLPack
    unsafe {
        common::take_deleter_and_delete(dlpack_f64);
        common::take_deleter_and_delete(dlpack_f32);
    }

    Some(min_fidelity)
}

#[test]
fn test_f32_vs_f64_amplitude_8_qubits() {
    let Some(fidelity) = compare_f32_f64_amplitude(8) else {
        println!("SKIP: No GPU available");
        return;
    };
    println!("F32 vs F64 fidelity @ 8 qubits: {:.10}", fidelity);
    assert!(
        fidelity > 1.0 - 1e-3,
        "Fidelity too low at 8 qubits: {fidelity}"
    );
}

#[test]
fn test_f32_vs_f64_amplitude_12_qubits() {
    let Some(fidelity) = compare_f32_f64_amplitude(12) else {
        println!("SKIP: No GPU available");
        return;
    };
    println!("F32 vs F64 fidelity @ 12 qubits: {:.10}", fidelity);
    assert!(
        fidelity > 1.0 - 1e-3,
        "Fidelity too low at 12 qubits: {fidelity}"
    );
}

#[test]
fn test_f32_vs_f64_amplitude_16_qubits() {
    let Some(fidelity) = compare_f32_f64_amplitude(16) else {
        println!("SKIP: No GPU available");
        return;
    };
    println!("F32 vs F64 fidelity @ 16 qubits: {:.10}", fidelity);
    assert!(
        fidelity > 1.0 - 1e-3,
        "Fidelity too low at 16 qubits: {fidelity}"
    );
}

#[test]
fn test_f32_vs_f64_amplitude_20_qubits() {
    let Some(fidelity) = compare_f32_f64_amplitude(20) else {
        println!("SKIP: No GPU available");
        return;
    };
    println!("F32 vs F64 fidelity @ 20 qubits: {:.10}", fidelity);
    // At 20 qubits (1M elements), F32 norm precision degrades.
    // We use a relaxed threshold here to document the baseline.
    assert!(
        fidelity > 1.0 - 1e-2,
        "Fidelity too low at 20 qubits: {fidelity}"
    );
}

// ═════════════════════════════════════════════════════════════════════════
// Angle encoding: F32 vs F64 fidelity
// ═══════════════════════════════════════════════════════════════════════

fn compare_f32_f64_angle(num_qubits: usize) -> Option<f64> {
    let engine_f64 = common::qdp_engine_with_precision(Precision::Float64)?;
    let engine_f32 = common::qdp_engine_with_precision(Precision::Float32)?;
    let device = common::cuda_device()?;

    let state_dim = 1usize << num_qubits;
    let num_samples = 4;
    let sample_size = num_qubits; // angle encoding: one angle per qubit

    // Build deterministic angle data
    let data_f64: Vec<f64> = (0..num_samples * sample_size)
        .map(|i| (i as f64) * std::f64::consts::PI / (num_samples * sample_size) as f64)
        .collect();
    let data_f32: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();

    let dlpack_f64 = engine_f64
        .encode_batch(&data_f64, num_samples, sample_size, num_qubits, "angle")
        .expect("F64 angle encode_batch should succeed");

    let dlpack_f32 = engine_f32
        .encode_batch_f32(&data_f32, num_samples, sample_size, num_qubits, "angle")
        .expect("F32 angle encode_batch should succeed");

    let f64_tensor = unsafe { &(*dlpack_f64).dl_tensor };
    let f32_tensor = unsafe { &(*dlpack_f32).dl_tensor };

    let total_elements = num_samples * state_dim;
    let host_f64 =
        download_complex_f64(&device, f64_tensor.data as *const _, total_elements).unwrap();
    let host_f32 =
        download_complex_f32(&device, f32_tensor.data as *const _, total_elements).unwrap();

    let mut min_fidelity = 1.0_f64;
    for s in 0..num_samples {
        let offset_f64 = s * state_dim * 2;
        let offset_f32 = s * state_dim * 2;
        let sample_f64 = &host_f64[offset_f64..offset_f64 + state_dim * 2];
        let sample_f32 = &host_f32[offset_f32..offset_f32 + state_dim * 2];
        let f = fidelity_cross_precision(sample_f32, sample_f64).unwrap();
        if f < min_fidelity {
            min_fidelity = f;
        }
    }

    unsafe {
        common::take_deleter_and_delete(dlpack_f64);
        common::take_deleter_and_delete(dlpack_f32);
    }

    Some(min_fidelity)
}

#[test]
fn test_f32_vs_f64_angle_8_qubits() {
    let Some(fidelity) = compare_f32_f64_angle(8) else {
        println!("SKIP: No GPU available");
        return;
    };
    println!("Angle F32 vs F64 fidelity @ 8 qubits: {:.10}", fidelity);
    assert!(
        fidelity > 1.0 - 1e-3,
        "Angle fidelity too low at 8 qubits: {fidelity}"
    );
}

#[test]
fn test_f32_vs_f64_angle_12_qubits() {
    let Some(fidelity) = compare_f32_f64_angle(12) else {
        println!("SKIP: No GPU available");
        return;
    };
    println!("Angle F32 vs F64 fidelity @ 12 qubits: {:.10}", fidelity);
    assert!(
        fidelity > 1.0 - 1e-3,
        "Angle fidelity too low at 12 qubits: {fidelity}"
    );
}

#[test]
fn test_f32_vs_f64_angle_16_qubits() {
    let Some(fidelity) = compare_f32_f64_angle(16) else {
        println!("SKIP: No GPU available");
        return;
    };
    println!("Angle F32 vs F64 fidelity @ 16 qubits: {:.10}", fidelity);
    assert!(
        fidelity > 1.0 - 1e-3,
        "Angle fidelity too low at 16 qubits: {fidelity}"
    );
}
