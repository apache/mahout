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

//! `QDP_NPY_MAX_BYTES` environment-variable tests.
//!
//! These mutate the process environment, so they live in their own test
//! binary and run as ONE test function: no other test thread can observe a
//! half-set variable. Do not add more `#[test]` functions to this file;
//! extend `qdp_npy_max_bytes_env_override` instead.

use ndarray::Array2;
use qdp_core::MahoutError;
use qdp_core::io::read_numpy_batch;
use qdp_core::readers::{NPY_MAX_BYTES_ENV, NpyMemoryLimit};

fn set_env(value: Option<&str>) {
    // SAFETY: this binary has a single test function, so no other thread
    // reads or writes the environment concurrently.
    unsafe {
        match value {
            Some(v) => std::env::set_var(NPY_MAX_BYTES_ENV, v),
            None => std::env::remove_var(NPY_MAX_BYTES_ENV),
        }
    }
}

#[test]
fn qdp_npy_max_bytes_env_override() {
    // 4 x 8 float64 = 256 bytes of data.
    let array = Array2::from_shape_vec((4, 8), (0..32).map(f64::from).collect()).unwrap();
    let file = tempfile::Builder::new().suffix(".npy").tempfile().unwrap();
    ndarray_npy::write_npy(file.path(), &array).unwrap();
    let path = file.path();

    // Unset -> Auto; a 256-byte array fits on any machine running tests.
    set_env(None);
    assert_eq!(NpyMemoryLimit::from_env().unwrap(), NpyMemoryLimit::Auto);
    assert_eq!(read_numpy_batch(path).unwrap().1, 4);

    // A limit below the estimate refuses the file, and says why.
    set_env(Some("255"));
    let err = read_numpy_batch(path).unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, MahoutError::MemoryAllocation(_)), "{err:?}");
    assert!(msg.contains("256 B (256 bytes)"), "{msg}");
    assert!(msg.contains("255 B (255 bytes)"), "{msg}");

    // A suffixed limit at or above the estimate allows it.
    set_env(Some("1K"));
    assert_eq!(
        NpyMemoryLimit::from_env().unwrap(),
        NpyMemoryLimit::MaxBytes(1024)
    );
    assert_eq!(read_numpy_batch(path).unwrap().0.len(), 32);

    // "off" disables the guard.
    set_env(Some("off"));
    assert_eq!(
        NpyMemoryLimit::from_env().unwrap(),
        NpyMemoryLimit::Disabled
    );
    assert!(read_numpy_batch(path).is_ok());

    // A typo is an error, not a silently disabled guard.
    set_env(Some("16 gigs"));
    let err = read_numpy_batch(path).unwrap_err();
    assert!(err.to_string().contains(NPY_MAX_BYTES_ENV), "{err}");

    set_env(None);
}
