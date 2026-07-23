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

//! Tests for [`qdp_core::estimate_memory`].
//!
//! Byte values are derived from the memory model in issue #1429:
//!   cpu_prefetch_bytes = prefetch_depth * batch_size * sample_len * bytes_per_elem
//!   gpu_state_bytes    = 2 * batch_size * state_len * complex_bytes  (complex_bytes = 2 * bpe)

use qdp_core::{Dtype, Encoding, estimate_memory};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * 1024 * 1024;

/// Reference case A from the issue: 16-qubit amplitude f32, batch 64, prefetch depth 16.
#[test]
fn reference_case_a_amplitude_16q_f32() {
    let est = estimate_memory(Encoding::Amplitude, 16, 64, Dtype::Float32, 16).unwrap();

    // cpu: 16 * 64 * 2^16 * 4 = 268,435,456 (256 MiB)
    assert_eq!(est.cpu_prefetch_bytes, 268_435_456);
    assert_eq!(est.cpu_prefetch_bytes, 256 * MIB);

    // gpu: 2 * 64 * 2^16 * 8 = 67,108,864 (64 MiB)
    assert_eq!(est.gpu_state_bytes, 67_108_864);
    assert_eq!(est.gpu_state_bytes, 64 * MIB);

    // total: 335,544,320 (320 MiB)
    assert_eq!(est.total(), 335_544_320);
    assert_eq!(est.total(), 320 * MIB);
}

/// Reference case B from the issue: 20-qubit amplitude f32, batch 64, prefetch depth 1.
#[test]
fn reference_case_b_amplitude_20q_f32() {
    let est = estimate_memory(Encoding::Amplitude, 20, 64, Dtype::Float32, 1).unwrap();

    // cpu: 1 * 64 * 2^20 * 4 = 268,435,456 (256 MiB)
    assert_eq!(est.cpu_prefetch_bytes, 268_435_456);
    assert_eq!(est.cpu_prefetch_bytes, 256 * MIB);

    // gpu: 2 * 64 * 2^20 * 8 = 1,073,741,824 (1 GiB)
    assert_eq!(est.gpu_state_bytes, 1_073_741_824);
    assert_eq!(est.gpu_state_bytes, GIB);

    // total: 1,342,177,280 (1.25 GiB)
    assert_eq!(est.total(), 1_342_177_280);
}

/// f64 doubles both terms of case A via `Dtype::bytes()` (8 vs 4).
#[test]
fn case_a_f64_doubles_both_terms() {
    let f64_est = estimate_memory(Encoding::Amplitude, 16, 64, Dtype::Float64, 16).unwrap();

    // cpu: 536,870,912 (512 MiB), gpu: 134,217,728 (128 MiB), total: 671,088,640 (640 MiB)
    assert_eq!(f64_est.cpu_prefetch_bytes, 536_870_912);
    assert_eq!(f64_est.gpu_state_bytes, 134_217_728);
    assert_eq!(f64_est.total(), 671_088_640);

    // Exactly double the f32 case A.
    let f32_est = estimate_memory(Encoding::Amplitude, 16, 64, Dtype::Float32, 16).unwrap();
    assert_eq!(f64_est.cpu_prefetch_bytes, 2 * f32_est.cpu_prefetch_bytes);
    assert_eq!(f64_est.gpu_state_bytes, 2 * f32_est.gpu_state_bytes);
    assert_eq!(f64_est.total(), 2 * f32_est.total());
}

#[test]
fn angle_and_basis_have_tiny_cpu_side() {
    // Angle input is one value per qubit; basis is a single index. The GPU state vector is 2^n
    // either way, so the device side dominates by orders of magnitude.
    let angle = estimate_memory(Encoding::Angle, 16, 64, Dtype::Float32, 4).unwrap();
    assert_eq!(angle.cpu_prefetch_bytes, 4 * 64 * 16 * 4);

    let basis = estimate_memory(Encoding::Basis, 16, 64, Dtype::Float32, 4).unwrap();
    // depth 4 * batch 64 * sample_len 1 * 4 bytes
    assert_eq!(basis.cpu_prefetch_bytes, 4 * 64 * 4);

    let amplitude = estimate_memory(Encoding::Amplitude, 16, 64, Dtype::Float32, 4).unwrap();
    assert_eq!(angle.gpu_state_bytes, amplitude.gpu_state_bytes);
    assert_eq!(basis.gpu_state_bytes, amplitude.gpu_state_bytes);
    assert!(angle.cpu_prefetch_bytes < amplitude.cpu_prefetch_bytes / 1000);
}

#[test]
fn iqp_f32_falls_back_to_f64() {
    // IQP has no f32 batch encode path, so a Float32 request must estimate as Float64 —
    // otherwise the estimate would report half the memory the pipeline really uses.
    let requested_f32 = estimate_memory(Encoding::Iqp, 30, 8, Dtype::Float32, 2).unwrap();
    let explicit_f64 = estimate_memory(Encoding::Iqp, 30, 8, Dtype::Float64, 2).unwrap();
    assert_eq!(requested_f32, explicit_f64);

    // vector_len(iqp, 30) = 30 + 30*29/2 = 465
    assert_eq!(requested_f32.cpu_prefetch_bytes, 2 * 8 * 465 * 8);
    assert_eq!(requested_f32.gpu_state_bytes, 2 * 8 * (1u64 << 30) * 16);
}

#[test]
fn amplitude_keeps_f32() {
    let f32_est = estimate_memory(Encoding::Amplitude, 12, 16, Dtype::Float32, 2).unwrap();
    let f64_est = estimate_memory(Encoding::Amplitude, 12, 16, Dtype::Float64, 2).unwrap();
    assert_ne!(f32_est, f64_est);
}

#[test]
fn oversized_qubit_count_errors_instead_of_panicking() {
    // 2^63 is representable but every downstream product overflows.
    assert!(estimate_memory(Encoding::Amplitude, 63, 64, Dtype::Float32, 1).is_err());
    // 2^64 is not representable at all.
    assert!(estimate_memory(Encoding::Amplitude, 64, 1, Dtype::Float32, 1).is_err());
    assert!(estimate_memory(Encoding::Angle, 200, 1, Dtype::Float32, 1).is_err());
}

#[test]
fn oversized_batch_errors_instead_of_panicking() {
    assert!(estimate_memory(Encoding::Amplitude, 30, usize::MAX, Dtype::Float64, 1).is_err());
    assert!(estimate_memory(Encoding::Amplitude, 20, 1 << 40, Dtype::Float64, 1024).is_err());
}

#[test]
fn zero_batch_size_estimates_zero() {
    let est = estimate_memory(Encoding::Amplitude, 16, 0, Dtype::Float32, 4).unwrap();
    assert_eq!(est.cpu_prefetch_bytes, 0);
    assert_eq!(est.gpu_state_bytes, 0);
    assert_eq!(est.total(), 0);
}
