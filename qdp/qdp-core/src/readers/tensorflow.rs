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

//! TensorFlow TensorProto format reader implementation.

use crate::error::{MahoutError, Result};
use crate::reader::DataReader;
use bytes::Bytes;
use prost::Message;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Reader for TensorFlow TensorProto files.
///
/// Supports Float64 tensors with shape [batch_size, feature_size] or [n].
/// Prefers tensor_content for efficient parsing, but still requires one copy to Vec<f64>.
///
/// # Byte Order
/// This implementation assumes little-endian byte order, which is the standard
/// on x86_64 platforms. TensorFlow typically uses host byte order.
pub struct TensorFlowReader {
    // Store either raw bytes or f64 values to avoid unnecessary conversions
    payload: TensorPayload,
    num_samples: usize,
    sample_size: usize,
    read: bool,
}

enum TensorPayload {
    Bytes(Bytes),
    F64(Vec<f64>),
}

impl TensorFlowReader {
    /// Create a new TensorFlow reader from a file path.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Read entire file into memory (single read to avoid multiple I/O operations)
        let mut file = File::open(path.as_ref()).map_err(|e| MahoutError::IoWithSource {
            message: format!("Failed to open TensorFlow file: {}", e),
            source: e,
        })?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| MahoutError::IoWithSource {
                message: format!("Failed to read TensorFlow file: {}", e),
                source: e,
            })?;

        // Use Bytes for decode input; with build.rs config.bytes(...) this avoids copying tensor_content during decode
        let buffer = Bytes::from(buffer);

        // Parse TensorProto
        let mut tensor_proto = crate::tf_proto::tensorflow::TensorProto::decode(buffer)
            .map_err(|e| MahoutError::Io(format!("Failed to parse TensorProto: {}", e)))?;

        // Validate dtype == DT_DOUBLE (2)
        // Official TensorFlow: DT_DOUBLE = 2 (not 9)
        const DT_DOUBLE: i32 = 2;
        if tensor_proto.dtype != DT_DOUBLE {
            return Err(MahoutError::InvalidInput(format!(
                "Expected DT_DOUBLE (2), got {}",
                tensor_proto.dtype
            )));
        }

        // Parse shape
        let shape = tensor_proto.tensor_shape.as_ref().ok_or_else(|| {
            MahoutError::InvalidInput("TensorProto.tensor_shape is missing".into())
        })?;
        let (num_samples, sample_size) = Self::parse_shape(shape)?;

        // Extract data (prefer tensor_content, fallback to double_val)
        // Check for integer overflow
        let expected_elems = num_samples.checked_mul(sample_size).ok_or_else(|| {
            MahoutError::InvalidInput(format!(
                "Tensor shape too large: {} * {} would overflow",
                num_samples, sample_size
            ))
        })?;
        let expected_bytes = expected_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                MahoutError::InvalidInput(format!(
                    "Tensor size too large: {} elements * {} bytes would overflow",
                    expected_elems,
                    std::mem::size_of::<f64>()
                ))
            })?;
        let payload = Self::extract_payload(&mut tensor_proto, expected_elems, expected_bytes)?;

        Ok(Self {
            payload,
            num_samples,
            sample_size,
            read: false,
        })
    }

    /// Parse shape, supports 1D and 2D tensors
    fn parse_shape(
        shape: &crate::tf_proto::tensorflow::TensorShapeProto,
    ) -> Result<(usize, usize)> {
        if shape.unknown_rank {
            return Err(MahoutError::InvalidInput(
                "Unsupported tensor shape: unknown_rank=true".into(),
            ));
        }

        let dims = &shape.dim;

        match dims.len() {
            1 => {
                // 1D: [n] -> single sample
                let size = dims[0].size;
                if size <= 0 {
                    return Err(MahoutError::InvalidInput(format!(
                        "Invalid dimension size: {}",
                        size
                    )));
                }
                Ok((1, size as usize))
            }
            2 => {
                // 2D: [batch_size, feature_size]
                let batch_size = dims[0].size;
                let feature_size = dims[1].size;
                if batch_size <= 0 || feature_size <= 0 {
                    return Err(MahoutError::InvalidInput(format!(
                        "Invalid shape: [{}, {}]",
                        batch_size, feature_size
                    )));
                }
                Ok((batch_size as usize, feature_size as usize))
            }
            _ => Err(MahoutError::InvalidInput(format!(
                "Unsupported tensor rank: {} (only 1D and 2D supported)",
                dims.len()
            ))),
        }
    }

    /// Safely extract tensor_content, handling alignment and byte order
    ///
    /// Prefers tensor_content (efficient parsing), falls back to double_val if unavailable.
    fn extract_payload(
        tensor_proto: &mut crate::tf_proto::tensorflow::TensorProto,
        expected_elems: usize,
        expected_bytes: usize,
    ) -> Result<TensorPayload> {
        if !tensor_proto.tensor_content.is_empty() {
            let content = std::mem::take(&mut tensor_proto.tensor_content);
            if content.len() != expected_bytes {
                return Err(MahoutError::InvalidInput(format!(
                    "tensor_content size mismatch: expected {} bytes, got {}",
                    expected_bytes,
                    content.len()
                )));
            }
            // With build.rs config.bytes(...), this is Bytes (avoids copy during decode)
            Ok(TensorPayload::Bytes(content))
        } else if !tensor_proto.double_val.is_empty() {
            let values = std::mem::take(&mut tensor_proto.double_val);
            if values.len() != expected_elems {
                return Err(MahoutError::InvalidInput(format!(
                    "double_val length mismatch: expected {} values, got {}",
                    expected_elems,
                    values.len()
                )));
            }
            Ok(TensorPayload::F64(values))
        } else {
            Err(MahoutError::InvalidInput(
                "TensorProto has no data (both tensor_content and double_val are empty)"
                    .to_string(),
            ))
        }
    }

    /// Convert `tensor_content` bytes to `Vec<f64>`.
    ///
    /// Note: Even though `tensor_content` can be zero-copy, `DataReader` requires `Vec<f64>`,
    /// so one copy is still needed. Uses memcpy (instead of element-wise `from_le_bytes`) for best performance.
    ///
    /// # Safety
    /// This function uses `unsafe` for memory copy, but performs the following safety checks:
    /// 1. Byte order check (little-endian only)
    /// 2. Length check (must be multiple of 8)
    /// 3. Alignment check (f64 needs 8-byte alignment, Vec handles this automatically)
    /// 4. Overflow check (ensures no overflow)
    fn bytes_to_f64_vec(bytes: &Bytes) -> Result<Vec<f64>> {
        if !cfg!(target_endian = "little") {
            return Err(MahoutError::NotImplemented(
                "Big-endian platforms are not supported for TensorFlow tensor_content".into(),
            ));
        }
        if !bytes.len().is_multiple_of(8) {
            return Err(MahoutError::InvalidInput(format!(
                "tensor_content length {} is not a multiple of 8",
                bytes.len()
            )));
        }

        let n = bytes.len() / 8;
        // Check overflow: ensure n doesn't exceed Vec's maximum capacity
        if n > (usize::MAX / std::mem::size_of::<f64>()) {
            return Err(MahoutError::InvalidInput(
                "tensor_content too large: would exceed maximum vector size".into(),
            ));
        }

        let mut data = Vec::<f64>::with_capacity(n);
        unsafe {
            // Safety: We've checked:
            // 1. bytes.len() % 8 == 0 (ensures divisible)
            // 2. n <= usize::MAX / size_of::<f64>() (ensures no overflow)
            // 3. Vec::with_capacity(n) ensures alignment (Rust Vec guarantees this)
            // 4. copy_nonoverlapping is safe because source and destination don't overlap
            // 5. Copy data first, then set length, ensuring memory is initialized
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                data.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
            data.set_len(n);
        }
        Ok(data)
    }
}

impl DataReader for TensorFlowReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        if self.read {
            return Err(MahoutError::InvalidInput(
                "Reader already consumed".to_string(),
            ));
        }
        self.read = true;

        match std::mem::replace(&mut self.payload, TensorPayload::F64(Vec::new())) {
            TensorPayload::F64(data) => {
                // Already Vec<f64>, return directly
                Ok((data, self.num_samples, self.sample_size))
            }
            TensorPayload::Bytes(bytes) => {
                let data = Self::bytes_to_f64_vec(&bytes)?;
                Ok((data, self.num_samples, self.sample_size))
            }
        }
    }

    fn get_sample_size(&self) -> Option<usize> {
        Some(self.sample_size)
    }

    fn get_num_samples(&self) -> Option<usize> {
        Some(self.num_samples)
    }
}
