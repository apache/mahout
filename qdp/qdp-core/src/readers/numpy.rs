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
//!
//! # Memory guard
//!
//! `.npy` files are loaded whole into host memory. Before the array body is
//! read, the reader parses only the header (shape, dtype, order), estimates
//! the in-memory size as `prod(shape) * itemsize`, and refuses the load if it
//! would not fit. See [`NpyMemoryLimit`] for the policy and the
//! [`NPY_MAX_BYTES_ENV`] environment variable for the override.

use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use std::str::FromStr;

use ndarray::Array2;
use ndarray_npy::{ReadNpyError, ReadNpyExt};
use py_literal::Value as PyValue;

use crate::error::{MahoutError, Result};
use crate::reader::DataReader;

/// Environment variable that overrides the host-memory guard for `.npy` loads.
///
/// Accepted values (case-insensitive, surrounding whitespace ignored):
/// - unset, empty or `auto`: the default dynamic limit ([`NpyMemoryLimit::Auto`]).
/// - a byte count such as `17179869184`, or one with a binary suffix
///   `K`, `M`, `G`, `T` (optionally followed by `B` or `iB`), e.g. `16G`,
///   `512MiB`: refuse arrays whose estimated size exceeds it
///   ([`NpyMemoryLimit::MaxBytes`]).
/// - `off`, `none`, `unlimited` or `disabled`: skip the check
///   ([`NpyMemoryLimit::Disabled`]).
///
/// Any other value is rejected with [`MahoutError::InvalidInput`] so a typo
/// cannot silently disable or loosen the guard.
pub const NPY_MAX_BYTES_ENV: &str = "QDP_NPY_MAX_BYTES";

/// Multiplier applied to the estimated array size in [`NpyMemoryLimit::Auto`]
/// mode. Loading can briefly hold two copies of the data (the parsed array and
/// the flattened row-major `Vec<f64>` for Fortran-order files, or the f64 data
/// plus a cast copy further down the pipeline).
pub const NPY_AUTO_SAFETY_FACTOR: u64 = 2;

/// Fraction of `MemAvailable` that [`NpyMemoryLimit::Auto`] is willing to use,
/// expressed as `NUMERATOR / DENOMINATOR` (80%) to keep the arithmetic exact.
const NPY_AUTO_AVAILABLE_NUMERATOR: u128 = 4;
const NPY_AUTO_AVAILABLE_DENOMINATOR: u128 = 5;

/// Largest header dictionary this reader will allocate for. NumPy itself
/// refuses headers over 10,000 bytes by default; this is a generous cap that
/// stops a corrupt `HEADER_LEN` field from causing a multi-GiB allocation.
const NPY_MAX_HEADER_LEN: usize = 1 << 20;

const NPY_MAGIC: &[u8] = b"\x93NUMPY";

/// How much host memory a `.npy` load may use before it is refused.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum NpyMemoryLimit {
    /// Refuse when `estimated_bytes * 2 > MemAvailable * 0.8`.
    ///
    /// If available memory cannot be determined (non-Linux, or no
    /// `MemAvailable` in `/proc/meminfo`) the load is allowed and a warning is
    /// logged, preserving the behaviour from before the guard existed.
    #[default]
    Auto,
    /// Refuse when the estimated array size exceeds this many bytes.
    /// No safety factor is applied: the value is compared to the raw estimate.
    MaxBytes(u64),
    /// Do not check.
    Disabled,
}

impl NpyMemoryLimit {
    /// Resolve the limit from [`NPY_MAX_BYTES_ENV`].
    pub fn from_env() -> Result<Self> {
        match std::env::var(NPY_MAX_BYTES_ENV) {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::Auto),
            Err(std::env::VarError::NotUnicode(_)) => Err(MahoutError::InvalidInput(format!(
                "{NPY_MAX_BYTES_ENV} is not valid UTF-8"
            ))),
        }
    }

    /// Parse a limit using the syntax documented on [`NPY_MAX_BYTES_ENV`].
    pub fn parse(value: &str) -> Result<Self> {
        let v = value.trim().to_ascii_lowercase();
        match v.as_str() {
            "" | "auto" => return Ok(Self::Auto),
            "off" | "none" | "unlimited" | "disabled" => return Ok(Self::Disabled),
            _ => {}
        }
        let invalid = || {
            MahoutError::InvalidInput(format!(
                "Invalid {NPY_MAX_BYTES_ENV} value {value:?}: expected a byte count \
                 (e.g. 17179869184, 16G, 512MiB), 'auto', or 'off'"
            ))
        };
        let split = v.find(|c: char| !c.is_ascii_digit()).unwrap_or(v.len());
        let (digits, suffix) = v.split_at(split);
        let number: u64 = digits.parse().map_err(|_| invalid())?;
        let shift = match suffix.trim_start() {
            "" | "b" => 0,
            "k" | "kb" | "kib" => 10,
            "m" | "mb" | "mib" => 20,
            "g" | "gb" | "gib" => 30,
            "t" | "tb" | "tib" => 40,
            _ => return Err(invalid()),
        };
        let bytes = number.checked_mul(1u64 << shift).ok_or_else(invalid)?;
        if bytes == 0 {
            return Err(MahoutError::InvalidInput(format!(
                "Invalid {NPY_MAX_BYTES_ENV} value {value:?}: a limit of 0 bytes would \
                 reject every file; use 'off' to disable the check"
            )));
        }
        Ok(Self::MaxBytes(bytes))
    }
}

/// The parts of a `.npy` header needed to size the array before reading it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NpyHeaderInfo {
    /// NumPy dtype string, e.g. `"<f8"`.
    pub descr: String,
    /// Bytes per element derived from `descr` (8 for `<f8`).
    pub itemsize: u64,
    /// `true` for column-major (Fortran) layout.
    pub fortran_order: bool,
    /// Array dimensions as written in the header.
    pub shape: Vec<u64>,
    /// Offset of the first data byte from the start of the file.
    pub data_offset: u64,
}

impl NpyHeaderInfo {
    /// `prod(shape) * itemsize`, or [`MahoutError::InvalidInput`] on overflow.
    pub fn estimated_bytes(&self) -> Result<u64> {
        let overflow = || {
            MahoutError::InvalidInput(format!(
                "NumPy header shape {} with dtype '{}' overflows a 64-bit byte count; \
                 the file is corrupt or not a real array",
                format_shape(&self.shape),
                self.descr
            ))
        };
        self.shape
            .iter()
            .try_fold(self.itemsize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(overflow)
    }
}

/// Parse a `.npy` header (format versions 1.0, 2.0 and 3.0) without reading
/// the array body. On success the reader is positioned at the first data byte.
pub fn parse_npy_header<R: Read>(reader: &mut R) -> Result<NpyHeaderInfo> {
    let io_err = |e: std::io::Error| MahoutError::IoWithSource {
        message: format!("Failed to read NumPy header: {}", e),
        source: e,
    };
    let bad = |msg: String| MahoutError::InvalidInput(format!("Invalid NumPy header: {msg}"));

    let mut prefix = [0u8; 8];
    reader.read_exact(&mut prefix).map_err(io_err)?;
    if &prefix[..6] != NPY_MAGIC {
        return Err(bad("missing \\x93NUMPY magic string".into()));
    }
    let (major, minor) = (prefix[6], prefix[7]);
    let (header_len, len_field_bytes) = match major {
        1 => {
            let mut b = [0u8; 2];
            reader.read_exact(&mut b).map_err(io_err)?;
            (u16::from_le_bytes(b) as usize, 2u64)
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b).map_err(io_err)?;
            let len = usize::try_from(u32::from_le_bytes(b))
                .map_err(|_| bad("HEADER_LEN does not fit in usize".into()))?;
            (len, 4u64)
        }
        _ => return Err(bad(format!("unsupported format version {major}.{minor}"))),
    };
    if header_len > NPY_MAX_HEADER_LEN {
        return Err(bad(format!(
            "header length {header_len} exceeds the {NPY_MAX_HEADER_LEN}-byte cap"
        )));
    }

    let mut dict_bytes = vec![0u8; header_len];
    reader.read_exact(&mut dict_bytes).map_err(io_err)?;
    let dict_str = std::str::from_utf8(&dict_bytes)
        .map_err(|_| bad("header dictionary is not valid UTF-8".into()))?;
    let dict = PyValue::from_str(dict_str.trim_end())
        .map_err(|e| bad(format!("cannot parse header dictionary: {e}")))?;
    let PyValue::Dict(entries) = dict else {
        return Err(bad("header is not a dictionary".into()));
    };

    let (mut descr, mut fortran_order, mut shape) = (None, None, None);
    for (key, value) in entries {
        match key.as_string().map(String::as_str) {
            Some("descr") => match value {
                PyValue::String(s) => descr = Some(s),
                _ => return Err(bad("structured dtypes are not supported".into())),
            },
            Some("fortran_order") => match value {
                PyValue::Boolean(b) => fortran_order = Some(b),
                other => return Err(bad(format!("fortran_order is not a bool: {other}"))),
            },
            Some("shape") => {
                let dims = value
                    .as_tuple()
                    .ok_or_else(|| bad(format!("shape is not a tuple: {value}")))?
                    .iter()
                    .map(|d| d.as_integer().and_then(|i| u64::try_from(i).ok()))
                    .collect::<Option<Vec<u64>>>()
                    .ok_or_else(|| {
                        bad(format!("shape {value} has a negative or >64-bit dimension"))
                    })?;
                shape = Some(dims);
            }
            _ => return Err(bad(format!("unexpected key {key}"))),
        }
    }
    let descr = descr.ok_or_else(|| bad("missing 'descr'".into()))?;
    let fortran_order = fortran_order.ok_or_else(|| bad("missing 'fortran_order'".into()))?;
    let shape = shape.ok_or_else(|| bad("missing 'shape'".into()))?;
    let itemsize =
        descr_itemsize(&descr).ok_or_else(|| bad(format!("unsupported dtype '{descr}'")))?;

    Ok(NpyHeaderInfo {
        descr,
        itemsize,
        fortran_order,
        shape,
        data_offset: NPY_MAGIC.len() as u64 + 2 + len_field_bytes + header_len as u64,
    })
}

/// Bytes per element for a simple NumPy dtype string (`<f8`, `|u1`, `<U10`,
/// `<M8[ns]`, ...). Returns `None` for object or unrecognised dtypes.
fn descr_itemsize(descr: &str) -> Option<u64> {
    let body = descr.strip_prefix(['<', '>', '|', '=']).unwrap_or(descr);
    let mut chars = body.chars();
    let kind = chars.next()?;
    let digits = chars.as_str().split('[').next()?;
    let n: u64 = digits.parse().ok()?;
    match kind {
        'b' | 'i' | 'u' | 'f' | 'c' | 'S' | 'a' | 'V' | 'M' | 'm' => Some(n),
        'U' => n.checked_mul(4), // UCS-4 code points
        _ => None,
    }
}

/// Decide whether an array described by `header` may be loaded.
///
/// Pure: it allocates nothing and reads no system state. `available_host_bytes`
/// is what [`NpyMemoryLimit::Auto`] compares against (pass the live
/// `MemAvailable`, or `None` if unknown). Returns the estimated size on success.
///
/// # Errors
/// - [`MahoutError::InvalidInput`] if `prod(shape) * itemsize` overflows `u64`.
/// - [`MahoutError::MemoryAllocation`] if the estimate exceeds the limit. The
///   message gives the estimate, the limit, how to override it, and suggests
///   Parquet streaming.
pub fn check_npy_memory(
    header: &NpyHeaderInfo,
    limit: NpyMemoryLimit,
    available_host_bytes: Option<u64>,
) -> Result<u64> {
    let estimated = header.estimated_bytes()?;
    let what = format!(
        "array of shape {} and dtype '{}' needs an estimated {} ({} bytes)",
        format_shape(&header.shape),
        header.descr,
        format_bytes(estimated),
        estimated
    );
    let advice = format!(
        "Set {NPY_MAX_BYTES_ENV} to a larger byte count (e.g. {NPY_MAX_BYTES_ENV}=64G) or to \
         'off' to override. For datasets this large, convert to Parquet and use the streaming \
         reader (QuantumDataLoader.source_file(path, streaming=True))."
    );

    match limit {
        NpyMemoryLimit::Disabled => Ok(estimated),
        NpyMemoryLimit::MaxBytes(max) => {
            if estimated > max {
                Err(MahoutError::MemoryAllocation(format!(
                    "Refusing to load NumPy file: {what}, which exceeds the configured limit \
                     of {} ({max} bytes). {advice}",
                    format_bytes(max)
                )))
            } else {
                Ok(estimated)
            }
        }
        NpyMemoryLimit::Auto => {
            let Some(available) = available_host_bytes else {
                return Ok(estimated);
            };
            let required = u128::from(estimated) * u128::from(NPY_AUTO_SAFETY_FACTOR);
            let budget = u128::from(available) * NPY_AUTO_AVAILABLE_NUMERATOR
                / NPY_AUTO_AVAILABLE_DENOMINATOR;
            if required > budget {
                Err(MahoutError::MemoryAllocation(format!(
                    "Refusing to load NumPy file: {what}; with a {NPY_AUTO_SAFETY_FACTOR}x \
                     safety factor that is more than the limit of 80% of available host \
                     memory ({} of {} MemAvailable). {advice}",
                    format_bytes(budget as u64),
                    format_bytes(available)
                )))
            } else {
                Ok(estimated)
            }
        }
    }
}

/// Human-readable binary size, e.g. `1.50 GiB`.
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

/// Python-style tuple text, e.g. `(3, 4)` or `(5,)`.
fn format_shape(shape: &[u64]) -> String {
    match shape {
        [single] => format!("({single},)"),
        dims => format!(
            "({})",
            dims.iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
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
    path: std::path::PathBuf,
    read: bool,
    /// `None` means "resolve from [`NPY_MAX_BYTES_ENV`] at read time".
    memory_limit: Option<NpyMemoryLimit>,
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
                return Err(MahoutError::IoWithSource {
                    message: format!(
                        "Failed to check if NumPy file exists at {}: {}",
                        path.display(),
                        e
                    ),
                    source: e,
                });
            }
            Ok(true) => {}
        }

        Ok(Self {
            path: path.to_path_buf(),
            read: false,
            memory_limit: None,
        })
    }

    /// Use an explicit host-memory limit instead of reading
    /// [`NPY_MAX_BYTES_ENV`]. Useful for callers (and tests) that want a fixed
    /// policy regardless of the process environment.
    #[must_use]
    pub fn with_memory_limit(mut self, limit: NpyMemoryLimit) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// Parse the header, apply the memory guard, and check that the file is
    /// long enough to hold the array the header describes. Leaves `file`
    /// positioned at the start so the full parser can re-read the header.
    fn preflight(&self, file: &mut File) -> Result<()> {
        let header = parse_npy_header(file)?;

        let limit = match self.memory_limit {
            Some(limit) => limit,
            None => NpyMemoryLimit::from_env()?,
        };
        let available = match limit {
            NpyMemoryLimit::Auto => {
                let available = crate::platform::available_host_memory_bytes();
                if available.is_none() {
                    log::warn!(
                        "Could not determine available host memory; loading {} without a \
                         size check. Set {NPY_MAX_BYTES_ENV} to enforce a limit.",
                        self.path.display()
                    );
                }
                available
            }
            _ => None,
        };
        let estimated = check_npy_memory(&header, limit, available)?;

        // Cheap consistency check: a header that promises more data than the
        // file holds is corrupt or truncated. Catch it here, before the full
        // parser allocates `estimated` bytes only to hit EOF.
        let file_len = file
            .metadata()
            .map_err(|e| MahoutError::IoWithSource {
                message: format!("Failed to stat NumPy file: {}", e),
                source: e,
            })?
            .len();
        let expected_len = header.data_offset.saturating_add(estimated);
        if file_len < expected_len {
            return Err(MahoutError::InvalidInput(format!(
                "NumPy file {} is truncated: header describes shape {} ('{}'), which needs \
                 {} bytes of data after the {}-byte header, but the file is only {} bytes",
                self.path.display(),
                format_shape(&header.shape),
                header.descr,
                estimated,
                header.data_offset,
                file_len
            )));
        }

        file.rewind().map_err(|e| MahoutError::IoWithSource {
            message: format!("Failed to rewind NumPy file: {}", e),
            source: e,
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

        let mut file = File::open(&self.path).map_err(|e| MahoutError::IoWithSource {
            message: format!("Failed to read NumPy file: {}", e),
            source: e,
        })?;

        // Size the array from its header before allocating anything for it.
        self.preflight(&mut file)?;

        // Read the .npy file
        let array = Array2::<f64>::read_npy(file).map_err(|e| match e {
            ReadNpyError::Io(io_err) => MahoutError::IoWithSource {
                message: format!("Failed to read NumPy file: {}", io_err),
                source: io_err,
            },
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_npy::WriteNpyExt;
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

    /// Build a version-1.0 `.npy` header for `dict` (padded to 64 bytes).
    fn npy_v1_header(dict: &str) -> Vec<u8> {
        let mut body = dict.as_bytes().to_vec();
        let unpadded = 10 + body.len() + 1;
        body.resize(body.len() + (64 - unpadded % 64) % 64, b' ');
        body.push(b'\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(body.len() as u16).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn header(descr: &str, shape: &[u64]) -> NpyHeaderInfo {
        NpyHeaderInfo {
            descr: descr.to_string(),
            itemsize: descr_itemsize(descr).unwrap(),
            fortran_order: false,
            shape: shape.to_vec(),
            data_offset: 128,
        }
    }

    const GIB: u64 = 1 << 30;

    #[test]
    fn parse_header_matches_ndarray_npy_writer() {
        let array = Array2::<f64>::zeros((3, 5));
        let mut bytes = Vec::new();
        array.write_npy(&mut bytes).unwrap();
        let mut cursor = std::io::Cursor::new(&bytes);
        let h = parse_npy_header(&mut cursor).unwrap();
        assert_eq!(h.descr, "<f8");
        assert_eq!(h.itemsize, 8);
        assert!(!h.fortran_order);
        assert_eq!(h.shape, vec![3, 5]);
        assert_eq!(h.data_offset, cursor.position());
        assert_eq!(
            h.data_offset + h.estimated_bytes().unwrap(),
            bytes.len() as u64
        );
    }

    #[test]
    fn parse_header_v2_and_v3() {
        for major in [2u8, 3] {
            let dict = b"{'descr': '<f8', 'fortran_order': True, 'shape': (7,), }\n";
            let mut bytes = b"\x93NUMPY".to_vec();
            bytes.extend_from_slice(&[major, 0]);
            bytes.extend_from_slice(&(dict.len() as u32).to_le_bytes());
            bytes.extend_from_slice(dict);
            let h = parse_npy_header(&mut bytes.as_slice()).unwrap();
            assert!(h.fortran_order);
            assert_eq!(h.shape, vec![7]);
            assert_eq!(h.data_offset, 12 + dict.len() as u64);
        }
    }

    #[test]
    fn parse_header_accepts_huge_dimensions() {
        // A shape that does not fit in memory must still parse: that is the
        // whole point of checking it before reading.
        let bytes = npy_v1_header(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1000000000000, 10), }",
        );
        let h = parse_npy_header(&mut bytes.as_slice()).unwrap();
        assert_eq!(h.shape, vec![1_000_000_000_000, 10]);
        assert_eq!(h.estimated_bytes().unwrap(), 80_000_000_000_000);
    }

    #[test]
    fn parse_header_rejects_garbage() {
        let err = parse_npy_header(&mut &b"NOTNUMPYxxxxxxxx"[..]).unwrap_err();
        assert!(err.to_string().contains("magic"), "{err}");

        let bytes = npy_v1_header("{'descr': '<f8', 'fortran_order': False, 'shape': (-1, 2), }");
        let err = parse_npy_header(&mut bytes.as_slice()).unwrap_err();
        assert!(err.to_string().contains("negative"), "{err}");

        let bytes = npy_v1_header("{'descr': '|O', 'fortran_order': False, 'shape': (1, 2), }");
        let err = parse_npy_header(&mut bytes.as_slice()).unwrap_err();
        assert!(err.to_string().contains("unsupported dtype"), "{err}");

        let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        let err = parse_npy_header(&mut bytes.as_slice()).unwrap_err();
        assert!(err.to_string().contains("cap"), "{err}");
    }

    #[test]
    fn descr_itemsize_known_types() {
        assert_eq!(descr_itemsize("<f8"), Some(8));
        assert_eq!(descr_itemsize(">f4"), Some(4));
        assert_eq!(descr_itemsize("|u1"), Some(1));
        assert_eq!(descr_itemsize("<c16"), Some(16));
        assert_eq!(descr_itemsize("<U10"), Some(40));
        assert_eq!(descr_itemsize("<M8[ns]"), Some(8));
        assert_eq!(descr_itemsize("|O"), None);
    }

    #[test]
    fn estimated_bytes_overflow_is_an_error() {
        let h = header("<f8", &[1 << 40, 1 << 40]);
        match check_npy_memory(&h, NpyMemoryLimit::Disabled, None) {
            Err(MahoutError::InvalidInput(msg)) => assert!(msg.contains("overflows"), "{msg}"),
            other => panic!("expected overflow error, got {other:?}"),
        }
    }

    #[test]
    fn auto_limit_uses_safety_factor_and_80_percent() {
        // 10 GiB available -> budget 8 GiB -> largest array 4 GiB.
        let available = Some(10 * GIB);
        let four_gib = header("<f8", &[GIB / 8, 4]);
        assert_eq!(
            check_npy_memory(&four_gib, NpyMemoryLimit::Auto, available).unwrap(),
            4 * GIB
        );

        let just_over = header("<f8", &[GIB / 8 + 1, 4]);
        let err = check_npy_memory(&just_over, NpyMemoryLimit::Auto, available).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, MahoutError::MemoryAllocation(_)));
        assert!(msg.contains("4.00 GiB"), "{msg}");
        assert!(msg.contains("8.00 GiB of 10.00 GiB"), "{msg}");
        assert!(msg.contains(NPY_MAX_BYTES_ENV), "{msg}");
        assert!(msg.contains("Parquet"), "{msg}");
    }

    #[test]
    fn auto_limit_allows_when_available_memory_unknown() {
        let h = header("<f8", &[1 << 30, 1 << 20]);
        assert_eq!(
            check_npy_memory(&h, NpyMemoryLimit::Auto, None).unwrap(),
            8 << 50
        );
    }

    #[test]
    fn explicit_limit_and_disabled() {
        let h = header("<f8", &[1024, 128]); // 1 MiB
        assert!(check_npy_memory(&h, NpyMemoryLimit::MaxBytes(1 << 20), Some(0)).is_ok());

        let err = check_npy_memory(&h, NpyMemoryLimit::MaxBytes((1 << 20) - 1), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1.00 MiB (1048576 bytes)"), "{err}");
        assert!(err.contains("configured limit"), "{err}");

        // Disabled ignores even zero available memory.
        assert!(check_npy_memory(&h, NpyMemoryLimit::Disabled, Some(0)).is_ok());
    }

    #[test]
    fn parse_limit_values() {
        use NpyMemoryLimit::*;
        assert_eq!(NpyMemoryLimit::parse("").unwrap(), Auto);
        assert_eq!(NpyMemoryLimit::parse(" AUTO ").unwrap(), Auto);
        assert_eq!(NpyMemoryLimit::parse("off").unwrap(), Disabled);
        assert_eq!(NpyMemoryLimit::parse("Unlimited").unwrap(), Disabled);
        assert_eq!(NpyMemoryLimit::parse("4096").unwrap(), MaxBytes(4096));
        assert_eq!(NpyMemoryLimit::parse("16G").unwrap(), MaxBytes(16 * GIB));
        assert_eq!(
            NpyMemoryLimit::parse("512MiB").unwrap(),
            MaxBytes(512 << 20)
        );
        assert_eq!(NpyMemoryLimit::parse("2 tb").unwrap(), MaxBytes(2 << 40));
        for bad in ["0", "-1", "lots", "1.5G", "16X", "99999999999T"] {
            assert!(
                NpyMemoryLimit::parse(bad).is_err(),
                "{bad} should be rejected"
            );
        }
    }

    #[test]
    fn format_helpers() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.50 KiB");
        assert_eq!(format_bytes(80_000_000_000_000), "72.76 TiB");
        assert_eq!(format_shape(&[5]), "(5,)");
        assert_eq!(format_shape(&[3, 4]), "(3, 4)");
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
