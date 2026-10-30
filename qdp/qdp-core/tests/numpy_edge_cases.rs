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

//! Edge-case tests for the `.npy` reader (`NumpyReader` / `read_numpy_batch`).
//!
//! These run on the CPU only (no GPU needed). They pin down which `.npy`
//! files QDP accepts and which error users get otherwise, as documented in
//! `docs/qdp/numpy-input.md`.

use std::io::Write;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, ShapeBuilder};
use qdp_core::MahoutError;
use qdp_core::io::read_numpy_batch;
use qdp_core::reader::DataReader;
use qdp_core::readers::NumpyReader;
use tempfile::TempDir;

fn tmp_path(dir: &TempDir, name: &str) -> PathBuf {
    dir.path().join(name)
}

/// Read a file and return the `InvalidInput` message, panicking otherwise.
fn invalid_input_message(path: &Path) -> String {
    match read_numpy_batch(path) {
        Err(MahoutError::InvalidInput(msg)) => msg,
        Err(other) => panic!("expected InvalidInput, got {other:?}"),
        Ok(v) => panic!("expected an error, got Ok({v:?})"),
    }
}

/// Fortran (column-major) files must come back in row-major (C) order so
/// that each consecutive `sample_size` chunk is one sample.
#[test]
fn fortran_order_values_are_returned_row_major() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "fortran.npy");
    // Logical rows: [0, 1, 2] and [3, 4, 5]; stored column-major.
    let mut array = Array2::<f64>::zeros((2, 3).f());
    for ((r, c), v) in array.indexed_iter_mut() {
        *v = (r * 3 + c) as f64;
    }
    assert!(!array.is_standard_layout());
    ndarray_npy::write_npy(&path, &array).unwrap();

    let header = std::fs::read(&path).unwrap();
    let header = String::from_utf8_lossy(&header[..128]);
    assert!(header.contains("'fortran_order': True"), "{header}");

    let (data, n, s) = read_numpy_batch(&path).unwrap();
    assert_eq!((n, s), (2, 3));
    assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

/// Big-endian float64 (`>f8`) is converted to native values.
#[test]
fn big_endian_float64_is_accepted() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "be.npy");
    let values = [1.5f64, -2.0, 3.25, 4.0];
    // Hand-written NPY v1.0 file: magic, version, header length, header.
    let mut header = "{'descr': '>f8', 'fortran_order': False, 'shape': (2, 2), }".to_string();
    // Pad so magic(6)+version(2)+len(2)+header is a multiple of 64, ending in '\n'.
    let pad = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    assert_eq!((10 + header.len()) % 64, 0);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    for v in values {
        f.write_all(&v.to_be_bytes()).unwrap();
    }
    drop(f);

    let (data, n, s) = read_numpy_batch(&path).unwrap();
    assert_eq!((n, s), (2, 2));
    assert_eq!(data, values.to_vec());
}

/// float32 files are rejected (no silent up-cast); users must save float64.
#[test]
fn float32_file_is_rejected_with_descriptor_error() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "f32.npy");
    let array = Array2::<f32>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    ndarray_npy::write_npy(&path, &array).unwrap();

    let msg = invalid_input_message(&path);
    assert!(msg.contains("Failed to parse NumPy file"), "{msg}");
    assert!(msg.contains("descriptor"), "{msg}");
    assert!(msg.contains("<f4"), "{msg}");
}

/// Integer files are rejected too (e.g. basis indices saved as int64).
#[test]
fn int64_file_is_rejected_with_descriptor_error() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "i64.npy");
    let array = Array2::<i64>::from_shape_vec((2, 1), vec![0, 3]).unwrap();
    ndarray_npy::write_npy(&path, &array).unwrap();

    let msg = invalid_input_message(&path);
    assert!(msg.contains("descriptor"), "{msg}");
    assert!(msg.contains("<i8"), "{msg}");
}

/// Files must be 2D; 1D and 3D arrays report the dimension mismatch.
#[test]
fn non_2d_files_report_ndim() {
    let dir = TempDir::new().unwrap();

    let p1 = tmp_path(&dir, "d1.npy");
    ndarray_npy::write_npy(&p1, &Array1::from_vec(vec![1.0f64, 2.0])).unwrap();
    let msg = invalid_input_message(&p1);
    assert!(msg.contains("ndim 1"), "{msg}");

    let p3 = tmp_path(&dir, "d3.npy");
    ndarray_npy::write_npy(&p3, &Array3::<f64>::ones((1, 2, 2))).unwrap();
    let msg = invalid_input_message(&p3);
    assert!(msg.contains("ndim 3"), "{msg}");
}

/// A zero-width sample (shape `(n, 0)`) is rejected like `(0, n)`.
#[test]
fn zero_width_samples_are_rejected() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "zero_width.npy");
    ndarray_npy::write_npy(&path, &Array2::<f64>::zeros((3, 0))).unwrap();

    let msg = invalid_input_message(&path);
    assert!(msg.contains("Both dimensions must be > 0"), "{msg}");
}

/// A file cut short (e.g. interrupted copy) is reported, not zero-filled.
#[test]
fn truncated_file_is_rejected() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "truncated.npy");
    let array = Array2::<f64>::ones((4, 4));
    ndarray_npy::write_npy(&path, &array).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 8]).unwrap();

    let msg = invalid_input_message(&path);
    assert!(msg.contains("Failed to parse NumPy file"), "{msg}");
    assert!(msg.contains("EOF"), "{msg}");
}

/// A file that is not NPY at all fails while parsing the header.
#[test]
fn non_npy_file_is_rejected() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "garbage.npy");
    std::fs::write(&path, b"definitely not a numpy file").unwrap();

    let msg = invalid_input_message(&path);
    assert!(msg.contains("error parsing header"), "{msg}");
}

/// NaN / Inf values are read as-is; rejecting them is the encoder's job
/// (amplitude: zero/NaN/Inf norm; angle, phase, iqp: finite check).
#[test]
fn non_finite_values_are_passed_through() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "nan.npy");
    let array = Array2::from_shape_vec((1, 3), vec![f64::NAN, f64::INFINITY, 1.0]).unwrap();
    ndarray_npy::write_npy(&path, &array).unwrap();

    let (data, n, s) = read_numpy_batch(&path).unwrap();
    assert_eq!((n, s), (1, 3));
    assert!(data[0].is_nan());
    assert!(data[1].is_infinite());
    assert_eq!(data[2], 1.0);
}

/// A missing file is an I/O error raised by `NumpyReader::new`, before any
/// parsing, and the message names the path.
#[test]
fn missing_file_is_io_error_with_path() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "does_not_exist.npy");
    match NumpyReader::new(&path) {
        Err(MahoutError::Io(msg)) => {
            assert!(msg.contains("NumPy file not found"), "{msg}");
            assert!(msg.contains("does_not_exist.npy"), "{msg}");
        }
        Err(other) => panic!("expected Io, got {other:?}"),
        Ok(_) => panic!("expected an error"),
    }
}

/// The metadata getters do not read the header yet (documented limitation).
#[test]
fn metadata_getters_return_none_before_read() {
    let dir = TempDir::new().unwrap();
    let path = tmp_path(&dir, "meta.npy");
    ndarray_npy::write_npy(&path, &Array2::<f64>::ones((2, 4))).unwrap();
    let reader = NumpyReader::new(&path).unwrap();
    assert_eq!(reader.get_num_samples(), None);
    assert_eq!(reader.get_sample_size(), None);
}
