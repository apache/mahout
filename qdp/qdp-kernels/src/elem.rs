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

//! [`KernelElem`]: compile-time mapping from a host float type to the CUDA
//! launchers that consume it.
//!
//! Generic host code can write `T::ANGLE_ENCODE_BATCH` instead of choosing
//! between `launch_angle_encode_batch` and `launch_angle_encode_batch_f32`
//! by hand. The choice is resolved at compile time; the launchers and their
//! exported C symbol names are unchanged.

use crate::*;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A real scalar type (`f32` or `f64`) that has CUDA kernel launchers.
///
/// Sealed: only `f32` and `f64` implement it. Kernels that exist only in
/// double precision (IQP, phase) are deliberately absent from this trait.
pub trait KernelElem: sealed::Sealed + Copy + Send + Sync + 'static {
    /// Complex amplitude type written by this precision's kernels
    /// ([`CuDoubleComplex`] for `f64`, [`CuComplex`] for `f32`).
    type Complex: Copy + Send + Sync + 'static;

    /// Rust name of the scalar type: `"f64"` or `"f32"`.
    const NAME: &'static str;

    /// Suffix that turns a double-precision launcher name into this type's
    /// launcher name: `""` for `f64`, `"_f32"` for `f32`. The finite-value
    /// check is the one pair that predates the convention (`..._f64` /
    /// `..._f32`).
    const SUFFIX: &'static str;

    /// [`launch_amplitude_encode`] / [`launch_amplitude_encode_f32`].
    const AMPLITUDE_ENCODE: AmplitudeEncodeFn<Self>;
    /// [`launch_amplitude_encode_batch`] / [`launch_amplitude_encode_batch_f32`].
    const AMPLITUDE_ENCODE_BATCH: AmplitudeEncodeBatchFn<Self>;
    /// [`launch_l2_norm`] / [`launch_l2_norm_f32`].
    const L2_NORM: L2NormFn<Self>;
    /// [`launch_l2_norm_batch`] / [`launch_l2_norm_batch_f32`].
    const L2_NORM_BATCH: L2NormBatchFn<Self>;
    /// [`launch_basis_encode`] / [`launch_basis_encode_f32`]. The index
    /// argument is `usize` for both; only the output precision differs.
    const BASIS_ENCODE: BasisEncodeFn;
    /// [`launch_basis_encode_batch`] / [`launch_basis_encode_batch_f32`].
    const BASIS_ENCODE_BATCH: BasisEncodeBatchFn;
    /// [`launch_angle_encode`] / [`launch_angle_encode_f32`].
    const ANGLE_ENCODE: AngleEncodeFn<Self>;
    /// [`launch_angle_encode_batch`] / [`launch_angle_encode_batch_f32`].
    const ANGLE_ENCODE_BATCH: AngleEncodeBatchFn<Self>;
    /// [`launch_check_finite_batch_f64`] / [`launch_check_finite_batch_f32`].
    const CHECK_FINITE_BATCH: CheckFiniteBatchFn<Self>;
}

impl KernelElem for f64 {
    type Complex = CuDoubleComplex;
    const NAME: &'static str = "f64";
    const SUFFIX: &'static str = "";
    const AMPLITUDE_ENCODE: AmplitudeEncodeFn<f64> = launch_amplitude_encode;
    const AMPLITUDE_ENCODE_BATCH: AmplitudeEncodeBatchFn<f64> = launch_amplitude_encode_batch;
    const L2_NORM: L2NormFn<f64> = launch_l2_norm;
    const L2_NORM_BATCH: L2NormBatchFn<f64> = launch_l2_norm_batch;
    const BASIS_ENCODE: BasisEncodeFn = launch_basis_encode;
    const BASIS_ENCODE_BATCH: BasisEncodeBatchFn = launch_basis_encode_batch;
    const ANGLE_ENCODE: AngleEncodeFn<f64> = launch_angle_encode;
    const ANGLE_ENCODE_BATCH: AngleEncodeBatchFn<f64> = launch_angle_encode_batch;
    const CHECK_FINITE_BATCH: CheckFiniteBatchFn<f64> = launch_check_finite_batch_f64;
}

impl KernelElem for f32 {
    type Complex = CuComplex;
    const NAME: &'static str = "f32";
    const SUFFIX: &'static str = "_f32";
    const AMPLITUDE_ENCODE: AmplitudeEncodeFn<f32> = launch_amplitude_encode_f32;
    const AMPLITUDE_ENCODE_BATCH: AmplitudeEncodeBatchFn<f32> = launch_amplitude_encode_batch_f32;
    const L2_NORM: L2NormFn<f32> = launch_l2_norm_f32;
    const L2_NORM_BATCH: L2NormBatchFn<f32> = launch_l2_norm_batch_f32;
    const BASIS_ENCODE: BasisEncodeFn = launch_basis_encode_f32;
    const BASIS_ENCODE_BATCH: BasisEncodeBatchFn = launch_basis_encode_batch_f32;
    const ANGLE_ENCODE: AngleEncodeFn<f32> = launch_angle_encode_f32;
    const ANGLE_ENCODE_BATCH: AngleEncodeBatchFn<f32> = launch_angle_encode_batch_f32;
    const CHECK_FINITE_BATCH: CheckFiniteBatchFn<f32> = launch_check_finite_batch_f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_suffixes() {
        assert_eq!(<f64 as KernelElem>::NAME, "f64");
        assert_eq!(<f32 as KernelElem>::NAME, "f32");
        assert_eq!(<f64 as KernelElem>::SUFFIX, "");
        assert_eq!(<f32 as KernelElem>::SUFFIX, "_f32");
    }

    #[test]
    fn complex_type_matches_precision() {
        use std::mem::size_of;
        assert_eq!(
            size_of::<<f64 as KernelElem>::Complex>(),
            2 * size_of::<f64>()
        );
        assert_eq!(
            size_of::<<f32 as KernelElem>::Complex>(),
            2 * size_of::<f32>()
        );
    }

    /// The basis pair shares one fn-pointer type, so a swapped mapping
    /// would still type-check; pin it by address. (Typed pairs cannot be
    /// swapped: `AngleEncodeFn<f32>` will not accept the f64 launcher.)
    #[test]
    fn basis_launchers_map_to_their_own_symbol() {
        let f64_single: BasisEncodeFn = launch_basis_encode;
        let f32_single: BasisEncodeFn = launch_basis_encode_f32;
        let f64_batch: BasisEncodeBatchFn = launch_basis_encode_batch;
        let f32_batch: BasisEncodeBatchFn = launch_basis_encode_batch_f32;
        assert_eq!(
            <f64 as KernelElem>::BASIS_ENCODE as usize,
            f64_single as usize
        );
        assert_eq!(
            <f32 as KernelElem>::BASIS_ENCODE as usize,
            f32_single as usize
        );
        assert_eq!(
            <f64 as KernelElem>::BASIS_ENCODE_BATCH as usize,
            f64_batch as usize
        );
        assert_eq!(
            <f32 as KernelElem>::BASIS_ENCODE_BATCH as usize,
            f32_batch as usize
        );
    }

    /// In a no-CUDA build every mapped launcher is a stub returning 999.
    #[test]
    #[cfg(qdp_no_cuda)]
    fn mapped_stubs_report_cuda_unavailable() {
        fn check<T: KernelElem>() {
            let n = std::ptr::null_mut();
            unsafe {
                assert_eq!(
                    (T::ANGLE_ENCODE_BATCH)(std::ptr::null(), n, 0, 0, 0, n),
                    999
                );
                assert_eq!((T::BASIS_ENCODE)(0, n, 0, n), 999);
                assert_eq!(
                    (T::CHECK_FINITE_BATCH)(std::ptr::null(), 0, std::ptr::null_mut(), n),
                    999
                );
            }
        }
        check::<f64>();
        check::<f32>();
    }
}
