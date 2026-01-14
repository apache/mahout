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

//! PyTorch tensor reader implementation.
//!
//! Supports `.pt`/`.pth` files containing a single tensor saved with `torch.save`.
//! The tensor must be 1D or 2D and will be converted to `float64`.
//! Requires the `pytorch` feature to be enabled.

use std::path::Path;

use crate::error::{MahoutError, Result};
use crate::reader::DataReader;

/// Reader for PyTorch `.pt`/`.pth` tensor files.
pub struct TorchReader {
    path: std::path::PathBuf,
    read: bool,
    num_samples: Option<usize>,
    sample_size: Option<usize>,
}

impl TorchReader {
    /// Create a new PyTorch reader.
    ///
    /// # Arguments
    /// * `path` - Path to the `.pt`/`.pth` file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        match path.try_exists() {
            Ok(false) => {
                return Err(MahoutError::Io(format!(
                    "PyTorch file not found: {}",
                    path.display()
                )));
            }
            Err(e) => {
                return Err(MahoutError::Io(format!(
                    "Failed to check if PyTorch file exists at {}: {}",
                    path.display(),
                    e
                )));
            }
            Ok(true) => {}
        }

        Ok(Self {
            path: path.to_path_buf(),
            read: false,
            num_samples: None,
            sample_size: None,
        })
    }
}

impl DataReader for TorchReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        if self.read {
            return Err(MahoutError::InvalidInput(
                "Reader already consumed".to_string(),
            ));
        }
        self.read = true;

        #[cfg(feature = "pytorch")]
        {
            let (data, num_samples, sample_size) = read_torch_tensor(&self.path)?;
            self.num_samples = Some(num_samples);
            self.sample_size = Some(sample_size);
            Ok((data, num_samples, sample_size))
        }

        #[cfg(not(feature = "pytorch"))]
        {
            Err(MahoutError::NotImplemented(
                "PyTorch reader requires the 'pytorch' feature".to_string(),
            ))
        }
    }

    fn get_sample_size(&self) -> Option<usize> {
        self.sample_size
    }

    fn get_num_samples(&self) -> Option<usize> {
        self.num_samples
    }
}

#[cfg(feature = "pytorch")]
fn read_torch_tensor(path: &Path) -> Result<(Vec<f64>, usize, usize)> {
    use tch::{Device, Kind, Tensor};

    let tensor = Tensor::load(path).map_err(|e| {
        MahoutError::Io(format!(
            "Failed to load PyTorch tensor from {}: {}",
            path.display(),
            e
        ))
    })?;

    let sizes = tensor.size();
    let (num_samples, sample_size) = parse_shape(&sizes)?;
    let tensor = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Double)
        .contiguous();

    let expected = num_samples.checked_mul(sample_size).ok_or_else(|| {
        MahoutError::InvalidInput(format!(
            "Tensor shape too large: {} * {} would overflow",
            num_samples, sample_size
        ))
    })?;

    let flat = tensor.view([-1]);
    let data: Vec<f64> = Vec::<f64>::try_from(&flat).map_err(|e| {
        MahoutError::InvalidInput(format!(
            "Failed to read PyTorch tensor data from {}: {}",
            path.display(),
            e
        ))
    })?;
    if data.len() != expected {
        return Err(MahoutError::InvalidInput(format!(
            "Tensor data length mismatch: expected {}, got {}",
            expected,
            data.len()
        )));
    }

    Ok((data, num_samples, sample_size))
}

#[cfg(feature = "pytorch")]
fn parse_shape(sizes: &[i64]) -> Result<(usize, usize)> {
    match sizes.len() {
        1 => {
            let sample_size = checked_dim(sizes[0], "sample")?;
            Ok((1, sample_size))
        }
        2 => {
            let num_samples = checked_dim(sizes[0], "batch")?;
            let sample_size = checked_dim(sizes[1], "feature")?;
            Ok((num_samples, sample_size))
        }
        _ => Err(MahoutError::InvalidInput(format!(
            "Unsupported tensor rank: {} (only 1D and 2D supported)",
            sizes.len()
        ))),
    }
}

#[cfg(feature = "pytorch")]
fn checked_dim(value: i64, label: &str) -> Result<usize> {
    if value <= 0 {
        return Err(MahoutError::InvalidInput(format!(
            "Invalid {} dimension size: {}",
            label, value
        )));
    }
    usize::try_from(value).map_err(|_| {
        MahoutError::InvalidInput(format!(
            "{} dimension too large to fit in usize: {}",
            label, value
        ))
    })
}
