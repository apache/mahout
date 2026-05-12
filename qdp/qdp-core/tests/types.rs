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

//! Tests for [`qdp_core::Encoding`] and [`qdp_core::Dtype`].

use qdp_core::{Backend, Dtype, Encoding};

#[test]
fn encoding_case_insensitive() {
    assert_eq!(
        Encoding::from_str_ci("Amplitude").unwrap(),
        Encoding::Amplitude
    );
    assert_eq!(
        Encoding::from_str_ci("AMPLITUDE").unwrap(),
        Encoding::Amplitude
    );
    assert_eq!(Encoding::from_str_ci("iqp-z").unwrap(), Encoding::IqpZ);
}

#[test]
fn encoding_unknown_returns_err() {
    assert!(Encoding::from_str_ci("not_real").is_err());
}

#[test]
fn vector_len_matches_encoder_contracts() {
    let n = 5u32;
    assert_eq!(Encoding::Amplitude.vector_len(n), 32); // 2^5
    assert_eq!(Encoding::Angle.vector_len(n), 5); // n
    assert_eq!(Encoding::IqpZ.vector_len(n), 5); // n (z-only)
    assert_eq!(Encoding::Phase.vector_len(n), 5); // n (one angle per qubit)
    assert_eq!(Encoding::Iqp.vector_len(n), 5 + 5 * 4 / 2); // n + n*(n-1)/2 = 15
    assert_eq!(Encoding::Basis.vector_len(n), 1);
}

#[test]
fn static_encoder_same_instance_across_calls() {
    assert!(
        std::ptr::eq(Encoding::Amplitude.encoder(), Encoding::Amplitude.encoder(),),
        "static dispatch must return the same 'static reference"
    );
}

#[test]
fn supports_f32_covers_amplitude_angle_basis() {
    // Amplitude / Angle / Basis have wired host f32 prefetch + encode_batch_f32 paths.
    // Iqp / IqpZ / Phase still normalize to Float64 until their batch f32 GPU paths exist.
    assert!(Encoding::Amplitude.supports_f32());
    assert!(Encoding::Angle.supports_f32());
    assert!(Encoding::Basis.supports_f32());
    assert!(!Encoding::Iqp.supports_f32());
    assert!(!Encoding::IqpZ.supports_f32());
    assert!(!Encoding::Phase.supports_f32());
}

#[test]
fn dtype_from_str_ci() {
    assert_eq!(Dtype::from_str_ci("f32").unwrap(), Dtype::Float32);
    assert_eq!(Dtype::from_str_ci("Float64").unwrap(), Dtype::Float64);
    assert!(Dtype::from_str_ci("bf16").is_err());
}

// ---- `Backend` enum (PR 1.5) ----

#[test]
fn backend_case_insensitive_and_aliases() {
    // Canonical names.
    assert_eq!(
        Backend::from_str_ci("cuda-nvidia").unwrap(),
        Backend::CudaNvidia
    );
    assert_eq!(
        Backend::from_str_ci("torch-ref").unwrap(),
        Backend::TorchRef
    );
    // Aliases.
    assert_eq!(Backend::from_str_ci("cuda").unwrap(), Backend::CudaNvidia);
    assert_eq!(Backend::from_str_ci("nvidia").unwrap(), Backend::CudaNvidia);
    assert_eq!(Backend::from_str_ci("CUDA").unwrap(), Backend::CudaNvidia);
    assert_eq!(Backend::from_str_ci("torch").unwrap(), Backend::TorchRef);
    assert_eq!(Backend::from_str_ci("torchref").unwrap(), Backend::TorchRef);
    assert_eq!(Backend::from_str_ci("cpu").unwrap(), Backend::TorchRef);
    // Whitespace tolerance.
    assert_eq!(
        Backend::from_str_ci("  cuda  ").unwrap(),
        Backend::CudaNvidia
    );
}

#[test]
fn backend_amd_rejected_with_helpful_message() {
    // AMD/Triton is the Python-side backend (#1158); the Rust core must reject
    // it explicitly so users don't think the Rust path silently ignored the flag.
    for name in ["amd", "AMD", "triton", "Triton", "triton-amd"] {
        let err = Backend::from_str_ci(name)
            .expect_err(&format!("backend '{}' should be rejected", name))
            .to_string();
        assert!(
            err.contains("AMD") && err.contains("Python"),
            "error for '{}' must mention AMD + the Python path, got: {}",
            name,
            err
        );
    }
}

#[test]
fn backend_unknown_rejected() {
    assert!(Backend::from_str_ci("metal").is_err());
    assert!(Backend::from_str_ci("opencl").is_err());
}

#[test]
fn backend_as_str_roundtrips() {
    assert_eq!(Backend::CudaNvidia.as_str(), "cuda-nvidia");
    assert_eq!(Backend::TorchRef.as_str(), "torch-ref");
    // Round-trip through `as_str` -> `from_str_ci` for every variant.
    for b in [Backend::CudaNvidia, Backend::TorchRef] {
        assert_eq!(Backend::from_str_ci(b.as_str()).unwrap(), b);
    }
}

#[test]
fn backend_detect_is_platform_appropriate() {
    let b = Backend::detect();
    #[cfg(target_os = "linux")]
    assert_eq!(b, Backend::CudaNvidia);
    #[cfg(not(target_os = "linux"))]
    assert_eq!(b, Backend::TorchRef);
}
