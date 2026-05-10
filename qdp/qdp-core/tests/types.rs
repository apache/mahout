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

use qdp_core::{Dtype, Encoding};

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
fn supports_f32_amplitude_only() {
    assert!(Encoding::Amplitude.supports_f32());
    assert!(!Encoding::Angle.supports_f32());
    assert!(!Encoding::Basis.supports_f32());
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
