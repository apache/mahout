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

//! Tests for [`qdp_core::reader::DataReader`], [`StreamingDataReader`], and [`FloatElem`].

use qdp_core::MahoutError;
use qdp_core::Result;
use qdp_core::reader::{DataReader, FloatElem, StreamingDataReader};

struct BatchReader<T: FloatElem> {
    data: Vec<T>,
    num_samples: usize,
    sample_size: usize,
    consumed: bool,
}

impl<T: FloatElem> DataReader<T> for BatchReader<T> {
    fn read_batch(&mut self) -> Result<(Vec<T>, usize, usize)> {
        if self.consumed {
            return Err(MahoutError::InvalidInput(
                "BatchReader already consumed".to_string(),
            ));
        }
        self.consumed = true;
        Ok((self.data.clone(), self.num_samples, self.sample_size))
    }
}

struct ChunkReader<T: FloatElem> {
    chunks: Vec<Vec<T>>,
    index: usize,
    total_rows: usize,
}

impl<T: FloatElem> DataReader<T> for ChunkReader<T> {
    fn read_batch(&mut self) -> Result<(Vec<T>, usize, usize)> {
        Err(MahoutError::InvalidInput(
            "ChunkReader supports streaming only".to_string(),
        ))
    }
}

impl<T: FloatElem> StreamingDataReader<T> for ChunkReader<T> {
    fn read_chunk(&mut self, buffer: &mut [T]) -> Result<usize> {
        if self.index >= self.chunks.len() {
            return Ok(0);
        }
        let chunk = &self.chunks[self.index];
        self.index += 1;
        let n = chunk.len().min(buffer.len());
        buffer[..n].copy_from_slice(&chunk[..n]);
        Ok(n)
    }

    fn total_rows(&self) -> usize {
        self.total_rows
    }
}

#[test]
fn data_reader_default_elem_type_is_f64() {
    let mut reader = BatchReader {
        data: vec![1.0, 2.0, 3.0, 4.0],
        num_samples: 2,
        sample_size: 2,
        consumed: false,
    };
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();
    assert_eq!(num_samples, 2);
    assert_eq!(sample_size, 2);
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn data_reader_f32_returns_vec_f32_without_widening() {
    let mut reader = BatchReader {
        data: vec![1.0f32, 2.0, 3.0, 4.0],
        num_samples: 2,
        sample_size: 2,
        consumed: false,
    };
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();
    assert_eq!(num_samples, 2);
    assert_eq!(sample_size, 2);
    assert_eq!(data, vec![1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn streaming_data_reader_f32_read_chunk() {
    let mut reader = ChunkReader {
        chunks: vec![vec![1.0f32, 2.0], vec![3.0, 4.0]],
        index: 0,
        total_rows: 2,
    };
    let mut buf = [0.0f32; 4];
    assert_eq!(reader.read_chunk(&mut buf[..2]).unwrap(), 2);
    assert_eq!(&buf[..2], &[1.0, 2.0]);
    assert_eq!(reader.read_chunk(&mut buf[2..]).unwrap(), 2);
    assert_eq!(&buf[2..], &[3.0, 4.0]);
    assert_eq!(reader.read_chunk(&mut buf).unwrap(), 0);
    assert_eq!(reader.total_rows(), 2);
}
