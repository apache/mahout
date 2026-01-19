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

//! NumPy format reader implementation.
//!
//! Provides support for reading .npy files containing 2D float64 arrays.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use ndarray::Array2;
use ndarray_npy::ReadNpyError;

use crate::error::{MahoutError, Result};
use crate::reader::{DataReader, StreamingDataReader};

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

#[derive(Clone, Debug)]
struct NpyHeader {
    fortran_order: bool,
    num_samples: usize,
    sample_size: usize,
    data_offset: u64,
    data_len_bytes: usize,
}

impl NpyHeader {
    fn total_elements(&self) -> usize {
        self.num_samples * self.sample_size
    }
}

fn parse_header_value<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let key_single = format!("'{}'", key);
    let mut start = header.find(&key_single);
    if start.is_none() {
        let key_double = format!("\"{}\"", key);
        start = header.find(&key_double);
    }
    let start = start.ok_or_else(|| {
        MahoutError::InvalidInput(format!("Missing '{}' entry in .npy header", key))
    })?;
    let rest = &header[start..];
    let colon = rest
        .find(':')
        .ok_or_else(|| MahoutError::InvalidInput("Malformed .npy header".to_string()))?;
    Ok(rest[colon + 1..].trim_start())
}

fn parse_quoted_value(header: &str, key: &str) -> Result<String> {
    let rest = parse_header_value(header, key)?;
    let mut chars = rest.chars();
    let quote = chars
        .next()
        .ok_or_else(|| MahoutError::InvalidInput("Malformed .npy header".to_string()))?;
    if quote != '\'' && quote != '"' {
        return Err(MahoutError::InvalidInput(format!(
            "Expected quoted value for '{}'",
            key
        )));
    }
    let rest = &rest[1..];
    let end = rest.find(quote).ok_or_else(|| {
        MahoutError::InvalidInput(format!("Unterminated string for '{}'", key))
    })?;
    Ok(rest[..end].to_string())
}

fn parse_bool_value(header: &str, key: &str) -> Result<bool> {
    let rest = parse_header_value(header, key)?;
    if rest.starts_with("True") {
        Ok(true)
    } else if rest.starts_with("False") {
        Ok(false)
    } else {
        Err(MahoutError::InvalidInput(format!(
            "Expected True/False for '{}'",
            key
        )))
    }
}

fn parse_shape_value(header: &str, key: &str) -> Result<Vec<usize>> {
    let rest = parse_header_value(header, key)?;
    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return Err(MahoutError::InvalidInput("Malformed shape in .npy header".to_string()));
    }
    let end = rest.find(')').ok_or_else(|| {
        MahoutError::InvalidInput("Malformed shape in .npy header".to_string())
    })?;
    let inner = &rest[1..end];
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let value = part.parse::<usize>().map_err(|e| {
            MahoutError::InvalidInput(format!("Invalid shape value '{}': {}", part, e))
        })?;
        dims.push(value);
    }
    if dims.is_empty() {
        return Err(MahoutError::InvalidInput(
            "Empty shape in .npy header".to_string(),
        ));
    }
    Ok(dims)
}

fn validate_descr(descr: &str) -> Result<()> {
    let (endian, typecode) = match descr.chars().next() {
        Some('<') | Some('>') | Some('|') | Some('=') => (Some(descr.chars().next().unwrap()), &descr[1..]),
        _ => (None, descr),
    };

    if typecode != "f8" {
        return Err(MahoutError::InvalidInput(format!(
            "Unsupported dtype '{}' in .npy file (expected f8)",
            descr
        )));
    }

    if let Some('>') = endian {
        return Err(MahoutError::InvalidInput(
            "Big-endian .npy files are not supported".to_string(),
        ));
    }

    if !cfg!(target_endian = "little") {
        return Err(MahoutError::InvalidInput(
            "NumPy .npy reader only supports little-endian hosts".to_string(),
        ));
    }

    Ok(())
}

fn read_npy_header(path: &Path, file: &mut File) -> Result<NpyHeader> {
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .map_err(|e| MahoutError::Io(format!("Failed to read NumPy header: {}", e)))?;
    if &magic != NPY_MAGIC {
        return Err(MahoutError::InvalidInput(
            "Invalid .npy file magic header".to_string(),
        ));
    }

    let mut version = [0u8; 2];
    file.read_exact(&mut version)
        .map_err(|e| MahoutError::Io(format!("Failed to read NumPy header: {}", e)))?;
    let major = version[0];
    let minor = version[1];

    let header_len = match major {
        1 => {
            let mut len_bytes = [0u8; 2];
            file.read_exact(&mut len_bytes)
                .map_err(|e| MahoutError::Io(format!("Failed to read NumPy header: {}", e)))?;
            u16::from_le_bytes(len_bytes) as usize
        }
        2 | 3 => {
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)
                .map_err(|e| MahoutError::Io(format!("Failed to read NumPy header: {}", e)))?;
            u32::from_le_bytes(len_bytes) as usize
        }
        _ => {
            return Err(MahoutError::InvalidInput(format!(
                "Unsupported .npy version {}.{}",
                major, minor
            )))
        }
    };

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| MahoutError::Io(format!("Failed to read NumPy header: {}", e)))?;
    let header_str = std::str::from_utf8(&header_bytes).map_err(|e| {
        MahoutError::InvalidInput(format!("Invalid .npy header encoding: {}", e))
    })?;

    let descr = parse_quoted_value(header_str, "descr")?;
    validate_descr(&descr)?;
    let fortran_order = parse_bool_value(header_str, "fortran_order")?;
    let shape = parse_shape_value(header_str, "shape")?;
    if shape.len() != 2 {
        return Err(MahoutError::InvalidInput(format!(
            "Expected 2D array, got {}D array with shape {:?}",
            shape.len(),
            shape
        )));
    }

    let num_samples = shape[0];
    let sample_size = shape[1];
    if num_samples == 0 || sample_size == 0 {
        return Err(MahoutError::InvalidInput(format!(
            "Invalid array shape: [{}, {}]. Both dimensions must be > 0",
            num_samples, sample_size
        )));
    }

    let total_elements = num_samples
        .checked_mul(sample_size)
        .ok_or_else(|| MahoutError::InvalidInput("Array size overflow".to_string()))?;
    let data_len_bytes = total_elements
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| MahoutError::InvalidInput("Array size overflow".to_string()))?;

    let data_offset = file
        .stream_position()
        .map_err(|e| MahoutError::Io(format!("Failed to read NumPy header: {}", e)))?;
    let file_len = file
        .metadata()
        .map_err(|e| MahoutError::Io(format!("Failed to stat NumPy file: {}", e)))?
        .len();
    if data_offset + data_len_bytes as u64 > file_len {
        return Err(MahoutError::InvalidInput(format!(
            "NumPy file {} is truncated (expected {} bytes of data)",
            path.display(),
            data_len_bytes
        )));
    }

    Ok(NpyHeader {
        fortran_order,
        num_samples,
        sample_size,
        data_offset,
        data_len_bytes,
    })
}

fn read_f64s_at(file: &mut File, offset: u64, out: &mut [f64]) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    if !cfg!(target_endian = "little") {
        return Err(MahoutError::InvalidInput(
            "NumPy .npy reader only supports little-endian hosts".to_string(),
        ));
    }
    let byte_len = out.len() * std::mem::size_of::<f64>();
    let out_bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len) };
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| MahoutError::Io(format!("Failed to seek NumPy file: {}", e)))?;
    file.read_exact(out_bytes)
        .map_err(|e| MahoutError::Io(format!("Failed to read NumPy data: {}", e)))?;
    Ok(())
}

fn copy_f64s_from_bytes(bytes: &[u8], out: &mut [f64]) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    if bytes.len() != out.len() * std::mem::size_of::<f64>() {
        return Err(MahoutError::InvalidInput(
            "Byte slice length does not match output buffer".to_string(),
        ));
    }
    if !cfg!(target_endian = "little") {
        return Err(MahoutError::InvalidInput(
            "NumPy .npy reader only supports little-endian hosts".to_string(),
        ));
    }

    let align = std::mem::align_of::<f64>();
    if (bytes.as_ptr() as usize) % align == 0 {
        let src = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, out.len()) };
        out.copy_from_slice(src);
        return Ok(());
    }

    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(chunk);
        out[i] = f64::from_le_bytes(arr);
    }
    Ok(())
}

/// Reader for NumPy `.npy` files containing 2D float64 arrays.
///
/// # Expected Format
/// - 2D array with shape `[num_samples, sample_size]`
/// - Data type: `float64`
/// - Fortran (column-major) or C (row-major) order supported
///
/// # Example
///
/// ```rust,ignore
/// use qdp_core::reader::DataReader;
/// use qdp_core::readers::NumpyReader;
///
/// let mut reader = NumpyReader::new("data.npy").unwrap();
/// let (data, num_samples, sample_size) = reader.read_batch().unwrap();
/// println!("Read {} samples of size {}", num_samples, sample_size);
/// ```
pub struct NumpyReader {
    path: PathBuf,
    read: bool,
}

impl NumpyReader {
    /// Create a new NumPy reader.
    ///
    /// # Arguments
    /// * `path` - Path to the `.npy` file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Verify file exists
        match path.try_exists() {
            Ok(false) => {
                return Err(MahoutError::Io(format!(
                    "NumPy file not found: {}",
                    path.display()
                )));
            }
            Err(e) => {
                return Err(MahoutError::Io(format!(
                    "Failed to check if NumPy file exists at {}: {}",
                    path.display(),
                    e
                )));
            }
            Ok(true) => {}
        }

        Ok(Self {
            path: path.to_path_buf(),
            read: false,
        })
    }
}

impl DataReader for NumpyReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        if self.read {
            return Err(MahoutError::InvalidInput(
                "Reader already consumed".to_string(),
            ));
        }
        self.read = true;

        // Read the .npy file
        let array: Array2<f64> = ndarray_npy::read_npy(&self.path).map_err(|e| match e {
            ReadNpyError::Io(io_err) => {
                MahoutError::Io(format!("Failed to read NumPy file: {}", io_err))
            }
            _ => MahoutError::InvalidInput(format!("Failed to parse NumPy file: {}", e)),
        })?;

        // Extract shape
        let shape = array.shape();
        if shape.len() != 2 {
            return Err(MahoutError::InvalidInput(format!(
                "Expected 2D array, got {}D array with shape {:?}",
                shape.len(),
                shape
            )));
        }

        let num_samples = shape[0];
        let sample_size = shape[1];

        if num_samples == 0 || sample_size == 0 {
            return Err(MahoutError::InvalidInput(format!(
                "Invalid array shape: [{}, {}]. Both dimensions must be > 0",
                num_samples, sample_size
            )));
        }

        // Flatten to Vec<f64>
        // Handle both C-contiguous (row-major) and Fortran-contiguous (column-major)
        let data = if array.is_standard_layout() {
            // C-contiguous: can use into_raw_vec_and_offset for zero-copy
            let (vec, offset) = array.into_raw_vec_and_offset();
            match offset {
                Some(off) if off > 0 => {
                    // If there's an offset, we need to copy
                    vec[off..].to_vec()
                }
                _ => vec,
            }
        } else {
            // Not C-contiguous: need to copy in row-major order
            let mut data = Vec::with_capacity(num_samples * sample_size);
            for row in array.rows() {
                data.extend(row.iter().copied());
            }
            data
        };

        Ok((data, num_samples, sample_size))
    }

    fn get_sample_size(&self) -> Option<usize> {
        // Could be determined by reading just the header
        // For now, return None as we read on demand
        None
    }

    fn get_num_samples(&self) -> Option<usize> {
        // Could be determined by reading just the header
        // For now, return None as we read on demand
        None
    }
}

/// Streaming reader for NumPy `.npy` files containing 2D float64 arrays.
///
/// Reads data in chunks without loading the entire file into memory.
pub struct NumpyStreamingReader {
    path: PathBuf,
    file: File,
    header: NpyHeader,
    row_cursor: usize,
    column_buf: Vec<f64>,
}

impl NumpyStreamingReader {
    /// Create a new streaming NumPy reader.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        match path.try_exists() {
            Ok(false) => {
                return Err(MahoutError::Io(format!(
                    "NumPy file not found: {}",
                    path.display()
                )));
            }
            Err(e) => {
                return Err(MahoutError::Io(format!(
                    "Failed to check if NumPy file exists at {}: {}",
                    path.display(),
                    e
                )));
            }
            Ok(true) => {}
        }

        let mut file = File::open(path)
            .map_err(|e| MahoutError::Io(format!("Failed to open NumPy file: {}", e)))?;
        let header = read_npy_header(path, &mut file)?;

        Ok(Self {
            path: path.to_path_buf(),
            file,
            header,
            row_cursor: 0,
            column_buf: Vec::new(),
        })
    }
}

impl DataReader for NumpyStreamingReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        let total_elements = self.header.total_elements();
        let mut data = vec![0.0; total_elements];
        let mut written = 0;
        while written < total_elements {
            let n = self.read_chunk(&mut data[written..])?;
            if n == 0 {
                break;
            }
            written += n;
        }
        if written != total_elements {
            data.truncate(written);
        }

        Ok((
            data,
            self.header.num_samples,
            self.header.sample_size,
        ))
    }

    fn get_sample_size(&self) -> Option<usize> {
        Some(self.header.sample_size)
    }

    fn get_num_samples(&self) -> Option<usize> {
        Some(self.header.num_samples)
    }
}

impl StreamingDataReader for NumpyStreamingReader {
    fn read_chunk(&mut self, buffer: &mut [f64]) -> Result<usize> {
        if self.row_cursor >= self.header.num_samples {
            return Ok(0);
        }

        let sample_size = self.header.sample_size;
        let max_rows = buffer.len() / sample_size;
        if max_rows == 0 {
            return Err(MahoutError::InvalidInput(format!(
                "Buffer too small for one sample (need {} elements)",
                sample_size
            )));
        }

        let remaining_rows = self.header.num_samples - self.row_cursor;
        let rows_to_read = std::cmp::min(max_rows, remaining_rows);
        let elem_count = rows_to_read * sample_size;

        if !self.header.fortran_order {
            let offset = self.header.data_offset
                + (self.row_cursor * sample_size * std::mem::size_of::<f64>()) as u64;
            read_f64s_at(&mut self.file, offset, &mut buffer[..elem_count])?;
        } else {
            if self.column_buf.len() < rows_to_read {
                self.column_buf.resize(rows_to_read, 0.0);
            }
            for col in 0..sample_size {
                let offset = self.header.data_offset
                    + ((col * self.header.num_samples + self.row_cursor)
                        * std::mem::size_of::<f64>()) as u64;
                let column = &mut self.column_buf[..rows_to_read];
                read_f64s_at(&mut self.file, offset, column)?;
                for row in 0..rows_to_read {
                    buffer[row * sample_size + col] = column[row];
                }
            }
        }

        self.row_cursor += rows_to_read;
        Ok(elem_count)
    }

    fn total_rows(&self) -> usize {
        self.header.num_samples
    }
}

/// Memory-mapped reader for NumPy `.npy` files containing 2D float64 arrays.
///
/// Maps the file into memory and streams slices without an extra read + flatten pass.
pub struct NumpyMmapReader {
    path: PathBuf,
    mmap: Mmap,
    header: NpyHeader,
    row_cursor: usize,
    column_buf: Vec<f64>,
}

impl NumpyMmapReader {
    /// Create a new memory-mapped NumPy reader.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        match path.try_exists() {
            Ok(false) => {
                return Err(MahoutError::Io(format!(
                    "NumPy file not found: {}",
                    path.display()
                )));
            }
            Err(e) => {
                return Err(MahoutError::Io(format!(
                    "Failed to check if NumPy file exists at {}: {}",
                    path.display(),
                    e
                )));
            }
            Ok(true) => {}
        }

        let mut file = File::open(path)
            .map_err(|e| MahoutError::Io(format!("Failed to open NumPy file: {}", e)))?;
        let header = read_npy_header(path, &mut file)?;
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| MahoutError::Io(format!("Failed to mmap NumPy file: {}", e)))?
        };

        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            header,
            row_cursor: 0,
            column_buf: Vec::new(),
        })
    }
}

impl DataReader for NumpyMmapReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        let total_elements = self.header.total_elements();
        let mut data = vec![0.0; total_elements];
        let mut written = 0;
        while written < total_elements {
            let n = self.read_chunk(&mut data[written..])?;
            if n == 0 {
                break;
            }
            written += n;
        }
        if written != total_elements {
            data.truncate(written);
        }

        Ok((
            data,
            self.header.num_samples,
            self.header.sample_size,
        ))
    }

    fn get_sample_size(&self) -> Option<usize> {
        Some(self.header.sample_size)
    }

    fn get_num_samples(&self) -> Option<usize> {
        Some(self.header.num_samples)
    }
}

impl StreamingDataReader for NumpyMmapReader {
    fn read_chunk(&mut self, buffer: &mut [f64]) -> Result<usize> {
        if self.row_cursor >= self.header.num_samples {
            return Ok(0);
        }

        let sample_size = self.header.sample_size;
        let max_rows = buffer.len() / sample_size;
        if max_rows == 0 {
            return Err(MahoutError::InvalidInput(format!(
                "Buffer too small for one sample (need {} elements)",
                sample_size
            )));
        }

        let remaining_rows = self.header.num_samples - self.row_cursor;
        let rows_to_read = std::cmp::min(max_rows, remaining_rows);
        let elem_count = rows_to_read * sample_size;
        let data_base = self.header.data_offset as usize;

        if !self.header.fortran_order {
            let start = data_base
                + self.row_cursor * sample_size * std::mem::size_of::<f64>();
            let end = start + elem_count * std::mem::size_of::<f64>();
            let bytes = &self.mmap[start..end];
            copy_f64s_from_bytes(bytes, &mut buffer[..elem_count])?;
        } else {
            if self.column_buf.len() < rows_to_read {
                self.column_buf.resize(rows_to_read, 0.0);
            }
            for col in 0..sample_size {
                let start = data_base
                    + (col * self.header.num_samples + self.row_cursor)
                        * std::mem::size_of::<f64>();
                let end = start + rows_to_read * std::mem::size_of::<f64>();
                let bytes = &self.mmap[start..end];
                let column = &mut self.column_buf[..rows_to_read];
                copy_f64s_from_bytes(bytes, column)?;
                for row in 0..rows_to_read {
                    buffer[row * sample_size + col] = column[row];
                }
            }
        }

        self.row_cursor += rows_to_read;
        Ok(elem_count)
    }

    fn total_rows(&self) -> usize {
        self.header.num_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::reader::StreamingDataReader;
    use std::fs;

    #[test]
    fn test_numpy_reader_basic() {
        // Create a test .npy file
        let temp_path = "/tmp/test_numpy_basic.npy";
        let num_samples = 5;
        let sample_size = 8;

        let mut data = Vec::with_capacity(num_samples * sample_size);
        for i in 0..num_samples {
            for j in 0..sample_size {
                data.push((i * sample_size + j) as f64);
            }
        }

        let array = Array2::from_shape_vec((num_samples, sample_size), data.clone()).unwrap();
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        // Read it back
        let mut reader = NumpyReader::new(temp_path).unwrap();
        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data.len(), num_samples * sample_size);
        assert_eq!(read_data, data);

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_reader_fortran_order() {
        // Create a Fortran-order (column-major) array
        let temp_path = "/tmp/test_numpy_fortran.npy";
        let num_samples = 3;
        let sample_size = 4;

        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();
        let array = Array2::from_shape_vec((num_samples, sample_size), data.clone()).unwrap();

        // Convert to Fortran order
        let array_f = array.reversed_axes();
        let array_f = array_f.as_standard_layout().reversed_axes();

        ndarray_npy::write_npy(temp_path, &array_f).unwrap();

        // Read it back
        let mut reader = NumpyReader::new(temp_path).unwrap();
        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data.len(), num_samples * sample_size);

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_streaming_reader_basic() {
        let temp_path = "/tmp/test_numpy_streaming_basic.npy";
        let num_samples = 4;
        let sample_size = 3;

        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();
        let array = Array2::from_shape_vec((num_samples, sample_size), data.clone()).unwrap();
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        let mut reader = NumpyStreamingReader::new(temp_path).unwrap();
        let mut buffer = vec![0.0; sample_size * 2];
        let mut out = Vec::new();
        loop {
            let written = reader.read_chunk(&mut buffer).unwrap();
            if written == 0 {
                break;
            }
            out.extend_from_slice(&buffer[..written]);
        }

        assert_eq!(out, data);
        assert_eq!(reader.total_rows(), num_samples);

        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_streaming_reader_fortran_order() {
        let temp_path = "/tmp/test_numpy_streaming_fortran.npy";
        let num_samples = 3;
        let sample_size = 4;

        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();
        let array = Array2::from_shape_vec((num_samples, sample_size), data.clone()).unwrap();
        let array_f = array.reversed_axes();
        let array_f = array_f.as_standard_layout().reversed_axes();
        ndarray_npy::write_npy(temp_path, &array_f).unwrap();

        let mut reader = NumpyStreamingReader::new(temp_path).unwrap();
        let mut buffer = vec![0.0; sample_size];
        let mut out = Vec::new();
        loop {
            let written = reader.read_chunk(&mut buffer).unwrap();
            if written == 0 {
                break;
            }
            out.extend_from_slice(&buffer[..written]);
        }

        assert_eq!(out, data);

        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_mmap_reader_basic() {
        let temp_path = "/tmp/test_numpy_mmap_basic.npy";
        let num_samples = 5;
        let sample_size = 2;

        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();
        let array = Array2::from_shape_vec((num_samples, sample_size), data.clone()).unwrap();
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        let mut reader = NumpyMmapReader::new(temp_path).unwrap();
        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data, data);

        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_mmap_reader_fortran_order() {
        let temp_path = "/tmp/test_numpy_mmap_fortran.npy";
        let num_samples = 2;
        let sample_size = 3;

        let data: Vec<f64> = (0..num_samples * sample_size).map(|i| i as f64).collect();
        let array = Array2::from_shape_vec((num_samples, sample_size), data.clone()).unwrap();
        let array_f = array.reversed_axes();
        let array_f = array_f.as_standard_layout().reversed_axes();
        ndarray_npy::write_npy(temp_path, &array_f).unwrap();

        let mut reader = NumpyMmapReader::new(temp_path).unwrap();
        let (read_data, read_samples, read_size) = reader.read_batch().unwrap();

        assert_eq!(read_samples, num_samples);
        assert_eq!(read_size, sample_size);
        assert_eq!(read_data, data);

        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_reader_file_not_found() {
        let result = NumpyReader::new("/tmp/nonexistent_numpy_file_12345.npy");
        assert!(result.is_err());
    }

    #[test]
    fn test_numpy_reader_invalid_dimensions() {
        // Create a 1D array (should fail)
        let temp_path = "/tmp/test_numpy_1d.npy";
        let array = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        let mut reader = NumpyReader::new(temp_path).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_reader_already_consumed() {
        let temp_path = "/tmp/test_numpy_consumed.npy";
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        let mut reader = NumpyReader::new(temp_path).unwrap();
        let _ = reader.read_batch().unwrap();

        // Second read should fail
        let result = reader.read_batch();
        assert!(result.is_err());

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_numpy_reader_empty_dimensions() {
        // Create an array with zero dimension
        let temp_path = "/tmp/test_numpy_empty.npy";
        let array = Array2::<f64>::zeros((0, 5));
        ndarray_npy::write_npy(temp_path, &array).unwrap();

        let mut reader = NumpyReader::new(temp_path).unwrap();
        let result = reader.read_batch();
        assert!(result.is_err());

        // Cleanup
        fs::remove_file(temp_path).unwrap();
    }
}
