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

// Throughput/latency pipeline using QdpEngine and encode_batch. Full loop runs in Rust;
// Python bindings release GIL during the run.

use std::f64::consts::PI;
use std::path::Path;
use std::time::Instant;

use crate::QdpEngine;
use crate::dlpack::DLManagedTensor;
use crate::error::{MahoutError, Result};
use crate::io;
use crate::reader::{NullHandling, StreamingDataReader};
use crate::readers::ParquetStreamingReader;

/// Configuration for throughput/latency pipeline runs (Python run_throughput_pipeline_py).
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub device_id: usize,
    pub num_qubits: u32,
    pub batch_size: usize,
    pub total_batches: usize,
    pub encoding_method: String,
    pub seed: Option<u64>,
    pub warmup_batches: usize,
    pub null_handling: NullHandling,
    pub float32_pipeline: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            num_qubits: 16,
            batch_size: 64,
            total_batches: 100,
            encoding_method: "amplitude".to_string(),
            seed: None,
            warmup_batches: 0,
            null_handling: NullHandling::FillZero,
            float32_pipeline: false,
        }
    }
}

/// Result of a throughput or latency pipeline run.
#[derive(Clone, Debug)]
pub struct PipelineRunResult {
    pub duration_sec: f64,
    pub vectors_per_sec: f64,
    pub latency_ms_per_vector: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BatchData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub struct PrefetchedBatch {
    pub data: BatchData,
    pub batch_n: usize,
    pub sample_size: usize,
    pub num_qubits: usize,
}

pub trait BatchProducer: Send + 'static {
    fn produce(&mut self) -> Result<Option<PrefetchedBatch>>;
}

pub struct SyntheticProducer {
    pub config: PipelineConfig,
    pub vector_len: usize,
    pub batch_index: usize,
    pub total_batches: usize,
    pub batch_buf: Vec<f64>,
}

impl SyntheticProducer {
    pub fn new(config: PipelineConfig, vector_len: usize) -> Self {
        let total_batches = config.total_batches;
        let batch_buf = vec![0.0; config.batch_size * vector_len];
        Self {
            config,
            vector_len,
            batch_index: 0,
            total_batches,
            batch_buf,
        }
    }
}

impl BatchProducer for SyntheticProducer {
    fn produce(&mut self) -> Result<Option<PrefetchedBatch>> {
        if self.batch_index >= self.total_batches {
            return Ok(None);
        }
        fill_batch_inplace(&self.config, self.batch_index, self.vector_len, &mut self.batch_buf);
        let data = if self.config.float32_pipeline {
            BatchData::F32(self.batch_buf.iter().map(|&v| v as f32).collect())
        } else {
            BatchData::F64(self.batch_buf.clone())
        };
        self.batch_index += 1;
        Ok(Some(PrefetchedBatch {
            data,
            batch_n: self.config.batch_size,
            sample_size: self.vector_len,
            num_qubits: self.config.num_qubits as usize,
        }))
    }
}

pub struct InMemoryProducer {
    pub data: Vec<f64>,
    pub cursor: usize,
    pub sample_size: usize,
    pub batch_size: usize,
    pub num_qubits: usize,
    pub batches_yielded: usize,
    pub batch_limit: usize,
}

impl BatchProducer for InMemoryProducer {
    fn produce(&mut self) -> Result<Option<PrefetchedBatch>> {
        if self.batches_yielded >= self.batch_limit {
            return Ok(None);
        }
        let remaining = (self.data.len() - self.cursor) / self.sample_size;
        if remaining == 0 {
            return Ok(None);
        }
        
        let batch_n = remaining.min(self.batch_size);
        let start = self.cursor;
        let end = start + batch_n * self.sample_size;
        self.cursor = end;
        self.batches_yielded += 1;
        let slice = self.data[start..end].to_vec();
        
        let data = BatchData::F64(slice);
        
        Ok(Some(PrefetchedBatch {
            data,
            batch_n,
            sample_size: self.sample_size,
            num_qubits: self.num_qubits,
        }))
    }
}

pub struct StreamingProducer {
    pub reader: ParquetStreamingReader,
    pub buffer: Vec<f64>,
    pub buffer_cursor: usize,
    pub read_chunk_scratch: Vec<f64>,
    pub sample_size: usize,
    pub batch_size: usize,
    pub num_qubits: usize,
    pub batches_yielded: usize,
    pub batch_limit: usize,
}

impl BatchProducer for StreamingProducer {
    fn produce(&mut self) -> Result<Option<PrefetchedBatch>> {
        if self.batches_yielded >= self.batch_limit {
            return Ok(None);
        }
        let required = self.batch_size * self.sample_size;
        while (self.buffer.len() - self.buffer_cursor) < required {
            let written = self.reader.read_chunk(&mut self.read_chunk_scratch)?;
            if written == 0 {
                break;
            }
            self.buffer.extend_from_slice(&self.read_chunk_scratch[..written]);
        }
        let available = self.buffer.len() - self.buffer_cursor;
        let available_samples = available / self.sample_size;
        
        if available_samples == 0 {
            return Ok(None);
        }
        
        let batch_n = available_samples.min(self.batch_size);
        let start = self.buffer_cursor;
        let end = start + batch_n * self.sample_size;
        self.buffer_cursor = end;
        self.batches_yielded += 1;
        let slice = self.buffer[start..end].to_vec();
        
        if self.buffer_cursor >= self.buffer.len() / BUFFER_COMPACT_DENOM {
            self.buffer.drain(..self.buffer_cursor);
            self.buffer_cursor = 0;
        }
        
        let data = BatchData::F64(slice);
        
        Ok(Some(PrefetchedBatch {
            data,
            batch_n,
            sample_size: self.sample_size,
            num_qubits: self.num_qubits,
        }))
    }
}

const DEFAULT_PREFETCH_DEPTH: usize = 16;

fn spawn_producer(
    mut producer: impl BatchProducer,
) -> (
    std::sync::mpsc::Receiver<Result<PrefetchedBatch>>,
    std::thread::JoinHandle<()>,
) {
    let (tx, rx) = std::sync::mpsc::sync_channel(DEFAULT_PREFETCH_DEPTH);
    let handle = std::thread::Builder::new()
        .name("qdp-prefetch".into())
        .spawn(move || {
            loop {
                match producer.produce() {
                    Ok(Some(batch)) => {
                        if tx.send(Ok(batch)).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
        })
        .expect("Failed to spawn prefetch thread");
    (rx, handle)
}

/// Default Parquet row group size for streaming reader (tunable).
const DEFAULT_PARQUET_ROW_GROUP_SIZE: usize = 2048;

/// When buffer_cursor >= buffer.len() / BUFFER_COMPACT_DENOM, compact by draining consumed prefix.
const BUFFER_COMPACT_DENOM: usize = 2;

/// Returns the path extension as lowercase ASCII (e.g. "parquet"), or None if missing/non-UTF8.
fn path_extension_lower(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
}

/// Dispatches by path extension to the appropriate io reader. Returns (data, num_samples, sample_size).
/// Unsupported or missing extension returns Err with message listing supported formats.
fn read_file_by_extension(
    path: &Path,
    null_handling: NullHandling,
) -> Result<(Vec<f64>, usize, usize)> {
    let ext_lower = path_extension_lower(path);
    let ext = ext_lower.as_deref();
    match ext {
        Some("parquet") => {
            use crate::reader::DataReader;
            let mut reader = crate::readers::ParquetReader::new(path, None, null_handling)?;
            reader.read_batch()
        }
        Some("arrow") | Some("feather") | Some("ipc") => {
            use crate::reader::DataReader;
            let mut reader = crate::readers::ArrowIPCReader::new(path, null_handling)?;
            reader.read_batch()
        }
        Some("npy") => io::read_numpy_batch(path),
        Some("pt") | Some("pth") => io::read_torch_batch(path),
        Some("pb") => io::read_tensorflow_batch(path),
        _ => Err(MahoutError::InvalidInput(format!(
            "Unsupported file extension {:?}. Supported: .parquet, .arrow, .feather, .ipc, .npy, .pt, .pth, .pb",
            path.extension()
        ))),
    }
}

/// Stateful iterator that yields one batch DLPack at a time for Python `for` loop consumption.
/// Reads prefetched batches via a bounded channel.
pub struct PipelineIterator {
    pub engine: QdpEngine,
    pub config: PipelineConfig,
    pub rx: std::sync::Mutex<std::sync::mpsc::Receiver<Result<PrefetchedBatch>>>,
    pub _producer_handle: std::sync::Mutex<std::thread::JoinHandle<()>>,
}

impl PipelineIterator {
    /// Create a new synthetic-data pipeline iterator.
    pub fn new_synthetic(engine: QdpEngine, config: PipelineConfig) -> Result<Self> {
        let vector_len = vector_len(config.num_qubits, &config.encoding_method);
        let producer = SyntheticProducer::new(config.clone(), vector_len);
        let (rx, _producer_handle) = spawn_producer(producer);
        Ok(Self {
            engine,
            config,
            rx: std::sync::Mutex::new(rx),
            _producer_handle: std::sync::Mutex::new(_producer_handle),
        })
    }

    /// Create a pipeline iterator from a file (Phase 2a: load full file then slice by batch).
    /// Dispatches by path extension; validates dimensions at construction.
    ///
    /// Supported extensions: .parquet, .arrow, .feather, .ipc, .npy, .pt, .pth, .pb.
    /// For file source, `batch_limit` caps batches yielded (e.g. for testing); use `usize::MAX` to iterate until EOF.
    pub fn new_from_file<P: AsRef<Path>>(
        engine: QdpEngine,
        path: P,
        config: PipelineConfig,
        batch_limit: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let (data, num_samples, sample_size) = read_file_by_extension(path, config.null_handling)?;
        let vector_len = vector_len(config.num_qubits, &config.encoding_method);

        // Dimension validation at construction.
        if sample_size != vector_len {
            return Err(MahoutError::InvalidInput(format!(
                "File feature length {} does not match vector_len {} for num_qubits={}, encoding={}",
                sample_size, vector_len, config.num_qubits, config.encoding_method
            )));
        }
        if data.len() != num_samples * sample_size {
            return Err(MahoutError::InvalidInput(format!(
                "File data length {} is not num_samples ({}) * sample_size ({})",
                data.len(),
                num_samples,
                sample_size
            )));
        }

        let producer = InMemoryProducer {
            data,
            cursor: 0,
            sample_size,
            batch_size: config.batch_size,
            num_qubits: config.num_qubits as usize,
            batches_yielded: 0,
            batch_limit,
        };
        let (rx, _producer_handle) = spawn_producer(producer);
        Ok(Self {
            engine,
            config,
            rx: std::sync::Mutex::new(rx),
            _producer_handle: std::sync::Mutex::new(_producer_handle),
        })
    }

    /// Create a pipeline iterator from a Parquet file using streaming read (Phase 2b).
    /// Only `.parquet` is supported; reduces memory for large files by reading in chunks.
    /// Validates sample_size == vector_len after the first chunk.
    pub fn new_from_file_streaming<P: AsRef<Path>>(
        engine: QdpEngine,
        path: P,
        config: PipelineConfig,
        batch_limit: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        if path_extension_lower(path).as_deref() != Some("parquet") {
            return Err(MahoutError::InvalidInput(format!(
                "Streaming file loader supports only .parquet; got extension {:?}. Use .source_file(path) for other formats.",
                path.extension()
            )));
        }

        let mut reader = ParquetStreamingReader::new(
            path,
            Some(DEFAULT_PARQUET_ROW_GROUP_SIZE),
            config.null_handling,
        )?;
        let vector_len = vector_len(config.num_qubits, &config.encoding_method);

        // Read first chunk to learn sample_size; reuse as initial buffer.
        const INITIAL_CHUNK_CAP: usize = 64 * 1024;
        let mut buffer = vec![0.0; INITIAL_CHUNK_CAP];
        let written = reader.read_chunk(&mut buffer)?;
        if written == 0 {
            return Err(MahoutError::InvalidInput(
                "Parquet file is empty or contains no data.".to_string(),
            ));
        }
        let sample_size = reader.get_sample_size().ok_or_else(|| {
            MahoutError::InvalidInput(
                "Parquet streaming reader did not set sample_size after first chunk.".to_string(),
            )
        })?;

        if sample_size != vector_len {
            return Err(MahoutError::InvalidInput(format!(
                "File feature length {} does not match vector_len {} for num_qubits={}, encoding={}",
                sample_size, vector_len, config.num_qubits, config.encoding_method
            )));
        }

        buffer.truncate(written);
        let read_chunk_scratch = vec![0.0; INITIAL_CHUNK_CAP];

        let producer = StreamingProducer {
            reader,
            buffer,
            buffer_cursor: 0,
            read_chunk_scratch,
            sample_size,
            batch_size: config.batch_size,
            num_qubits: config.num_qubits as usize,
            batches_yielded: 0,
            batch_limit,
        };
        let (rx, _producer_handle) = spawn_producer(producer);
        Ok(Self {
            engine,
            config,
            rx: std::sync::Mutex::new(rx),
            _producer_handle: std::sync::Mutex::new(_producer_handle),
        })
    }

    /// Returns the next batch as a DLPack pointer; `Ok(None)` when exhausted.
    pub fn next_batch(&mut self) -> Result<Option<*mut DLManagedTensor>> {
        let batch = match self.rx.lock().unwrap().recv() {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Ok(None),
        };
        let ptr = match batch.data {
            BatchData::F64(ref buf) => self.engine.encode_batch(
                buf,
                batch.batch_n,
                batch.sample_size,
                batch.num_qubits,
                &self.config.encoding_method,
            )?,
            BatchData::F32(ref buf) => self.engine.encode_batch_f32(
                buf,
                batch.batch_n,
                batch.sample_size,
                batch.num_qubits,
                &self.config.encoding_method,
            )?,
        };
        Ok(Some(ptr))
    }
}

/// Vector length per sample for given encoding (used by pipeline and iterator).
pub fn vector_len(num_qubits: u32, encoding_method: &str) -> usize {
    let n = num_qubits as usize;
    match encoding_method.to_lowercase().as_str() {
        "angle" => n,
        "basis" => 1,
        _ => 1 << n, // amplitude
    }
}

/// Deterministic sample generation matching Python utils.build_sample (amplitude/angle/basis).
fn fill_sample(seed: u64, out: &mut [f64], encoding_method: &str, num_qubits: usize) -> Result<()> {
    let len = out.len();
    if len == 0 {
        return Ok(());
    }
    match encoding_method.to_lowercase().as_str() {
        "basis" => {
            // For basis encoding, use 2^num_qubits as the state space size for mask calculation
            let state_space_size = 1 << num_qubits;
            let mask = (state_space_size - 1) as u64;
            let idx = seed & mask;
            out[0] = idx as f64;
        }
        "angle" => {
            let scale = (2.0 * PI) / len as f64;
            for (i, v) in out.iter_mut().enumerate() {
                let mixed = (i as u64 + seed) % (len as u64);
                *v = mixed as f64 * scale;
            }
        }
        _ => {
            // amplitude
            let mask = (len - 1) as u64;
            let scale = 1.0 / len as f64;
            for (i, v) in out.iter_mut().enumerate() {
                let mixed = (i as u64 + seed) & mask;
                *v = mixed as f64 * scale;
            }
        }
    }
    Ok(())
}

/// Generate one batch (batch_size * vector_len elements, or batch_size * 1 for basis).
#[cfg(test)]
fn generate_batch(config: &PipelineConfig, batch_idx: usize, vector_len: usize) -> Vec<f64> {
    let mut batch = vec![0.0f64; config.batch_size * vector_len];
    fill_batch_inplace(config, batch_idx, vector_len, &mut batch);
    batch
}

/// Fill an existing batch buffer in-place (avoids per-iteration allocations in benchmarks).
fn fill_batch_inplace(
    config: &PipelineConfig,
    batch_idx: usize,
    vector_len: usize,
    batch_buf: &mut [f64],
) {
    debug_assert_eq!(batch_buf.len(), config.batch_size * vector_len);
    let seed_base = config
        .seed
        .unwrap_or(0)
        .wrapping_add((batch_idx * config.batch_size) as u64);
    for i in 0..config.batch_size {
        let offset = i * vector_len;
        let _ = fill_sample(
            seed_base + i as u64,
            &mut batch_buf[offset..offset + vector_len],
            &config.encoding_method,
            config.num_qubits as usize,
        );
    }
}

/// Release DLPack tensor (call deleter so GPU memory is freed).
unsafe fn release_dlpack(ptr: *mut DLManagedTensor) {
    if ptr.is_null() {
        return;
    }
    let managed = unsafe { &mut *ptr };
    if let Some(deleter) = managed.deleter.take() {
        unsafe { deleter(ptr) };
    }
}

/// Run throughput pipeline: warmup, then timed encode_batch loop; returns stats.
pub fn run_throughput_pipeline(config: &PipelineConfig) -> Result<PipelineRunResult> {
    let engine = QdpEngine::new(config.device_id)?;
    let vector_len = vector_len(config.num_qubits, &config.encoding_method);
    let num_qubits = config.num_qubits as usize;

    let mut batch_buf = vec![0.0f64; config.batch_size * vector_len];

    // Warmup
    for b in 0..config.warmup_batches {
        fill_batch_inplace(config, b, vector_len, &mut batch_buf);
        let ptr = engine.encode_batch(
            &batch_buf,
            config.batch_size,
            vector_len,
            num_qubits,
            &config.encoding_method,
        )?;
        unsafe { release_dlpack(ptr) };
    }

    #[cfg(target_os = "linux")]
    engine.synchronize()?;

    let start = Instant::now();
    
    let producer = SyntheticProducer::new(config.clone(), vector_len);
    let (rx, producer_handle) = spawn_producer(producer);

    // Iteration loop
    let mut total_batches = 0;
    while let Ok(Ok(batch)) = rx.recv() {
        let ptr = match batch.data {
            BatchData::F64(ref buf) => engine.encode_batch(
                buf,
                batch.batch_n,
                batch.sample_size,
                batch.num_qubits,
                &config.encoding_method,
            )?,
            BatchData::F32(ref buf) => engine.encode_batch_f32(
                buf,
                batch.batch_n,
                batch.sample_size,
                batch.num_qubits,
                &config.encoding_method,
            )?,
        };
        unsafe { release_dlpack(ptr) };
        total_batches += 1;
    }

    let _ = producer_handle.join();

    #[cfg(target_os = "linux")]
    engine.synchronize()?;

    let duration_sec = start.elapsed().as_secs_f64().max(1e-9);
    let total_vectors = total_batches * config.batch_size;
    let vectors_per_sec = total_vectors as f64 / duration_sec;
    let latency_ms_per_vector = (duration_sec / total_vectors as f64) * 1000.0;

    Ok(PipelineRunResult {
        duration_sec,
        vectors_per_sec,
        latency_ms_per_vector,
    })
}

/// Run latency pipeline (same as throughput; returns same stats; name for API parity).
pub fn run_latency_pipeline(config: &PipelineConfig) -> Result<PipelineRunResult> {
    run_throughput_pipeline(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_generate_and_inplace_match(encoding_method: &str) {
        let config = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding_method: encoding_method.to_string(),
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, &config.encoding_method);

        // Test edge cases: 0 and batch_size-1
        for batch_idx in [0, config.batch_size - 1, 7] {
            let generated = generate_batch(&config, batch_idx, vector_len);
            let mut buf = vec![0.0f64; config.batch_size * vector_len];
            fill_batch_inplace(&config, batch_idx, vector_len, &mut buf);

            assert_eq!(generated, buf);
        }
    }

    fn assert_adjacent_batches_differ(encoding_method: &str) {
        let config = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding_method: encoding_method.to_string(),
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, &config.encoding_method);

        let batch0 = generate_batch(&config, 0, vector_len);
        let batch1 = generate_batch(&config, 1, vector_len);
        assert_ne!(batch0, batch1);
    }

    #[test]
    fn generate_batch_matches_fill_batch_inplace_amplitude() {
        assert_generate_and_inplace_match("amplitude");
    }

    #[test]
    fn generate_batch_matches_fill_batch_inplace_angle() {
        assert_generate_and_inplace_match("angle");
    }

    #[test]
    fn generate_batch_matches_fill_batch_inplace_basis() {
        assert_generate_and_inplace_match("basis");
    }

    #[test]
    fn adjacent_batches_differ_amplitude() {
        assert_adjacent_batches_differ("amplitude");
    }

    #[test]
    fn adjacent_batches_differ_angle() {
        assert_adjacent_batches_differ("angle");
    }

    #[test]
    fn adjacent_batches_differ_basis() {
        assert_adjacent_batches_differ("basis");
    }

    #[test]
    fn test_seed_none() {
        let config = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding_method: "amplitude".to_string(),
            seed: None,
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, &config.encoding_method);
        let batch = generate_batch(&config, 0, vector_len);
        assert_eq!(batch.len(), config.batch_size * vector_len);

        let mut buf = vec![0.0f64; config.batch_size * vector_len];
        fill_batch_inplace(&config, 0, vector_len, &mut buf);
        assert_eq!(batch, buf);
    }

    #[test]
    fn test_batch_size_one() {
        let config = PipelineConfig {
            num_qubits: 5,
            batch_size: 1,
            encoding_method: "amplitude".to_string(),
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, &config.encoding_method);
        let batch = generate_batch(&config, 0, vector_len);
        assert_eq!(batch.len(), vector_len);

        let mut buf = vec![0.0f64; vector_len];
        fill_batch_inplace(&config, 0, vector_len, &mut buf);
        assert_eq!(batch, buf);

        let batch0 = generate_batch(&config, 0, vector_len);
        let batch1 = generate_batch(&config, 1, vector_len);
        assert_ne!(batch0, batch1);
    }

    #[test]
    fn test_amplitude_encoding_case_insensitive() {
        let config_lower = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding_method: "amplitude".to_string(),
            seed: Some(123),
            ..Default::default()
        };

        let config_upper = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding_method: "AMPLITUDE".to_string(),
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config_lower.num_qubits, &config_lower.encoding_method);
        let batch_lower = generate_batch(&config_lower, 0, vector_len);
        let batch_upper = generate_batch(&config_upper, 0, vector_len);
        assert_eq!(batch_lower, batch_upper);
    }

    #[test]
    fn test_amplitude_samples_in_range() {
        let config = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding_method: "amplitude".to_string(),
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, &config.encoding_method);

        for batch_idx in 0..5 {
            let batch = generate_batch(&config, batch_idx, vector_len);
            for &value in &batch {
                assert!(
                    (0.0..1.0).contains(&value),
                    "amplitude value should be in [0, 1), got {} at batch_idx={}",
                    value,
                    batch_idx
                );
            }
        }
    }

    #[test]
    fn test_amplitude_samples_in_range_with_seed_none() {
        let config = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding_method: "amplitude".to_string(),
            seed: None,
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, &config.encoding_method);
        let batch = generate_batch(&config, 0, vector_len);

        for &value in &batch {
            assert!(
                (0.0..1.0).contains(&value),
                "amplitude value should be in [0, 1) with seed=None, got {}",
                value
            );
        }
    }

    #[test]
    fn test_amplitude_samples_in_range_batch_size_one() {
        let config = PipelineConfig {
            num_qubits: 5,
            batch_size: 1,
            encoding_method: "amplitude".to_string(),
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, &config.encoding_method);
        let batch = generate_batch(&config, 0, vector_len);

        for &value in &batch {
            assert!(
                (0.0..1.0).contains(&value),
                "amplitude value should be in [0, 1) with batch_size=1, got {}",
                value
            );
        }
    }
    #[test]
    fn test_synthetic_producer_batch_count() {
        let config = PipelineConfig {
            total_batches: 5,
            num_qubits: 3,
            batch_size: 4,
            encoding_method: "amplitude".to_string(),
            ..Default::default()
        };
        let vector_len = super::vector_len(config.num_qubits, &config.encoding_method);
        let mut producer = SyntheticProducer::new(config, vector_len);
        
        let mut count = 0;
        while let Ok(Some(_)) = producer.produce() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_synthetic_producer_data_consistency() {
        let config = PipelineConfig {
            total_batches: 1,
            num_qubits: 3,
            batch_size: 4,
            encoding_method: "amplitude".to_string(),
            ..Default::default()
        };
        let vector_len = super::vector_len(config.num_qubits, &config.encoding_method);
        let mut producer = SyntheticProducer::new(config.clone(), vector_len);
        
        let batch_from_producer = producer.produce().unwrap().unwrap();
        let expected_data = generate_batch(&config, 0, vector_len);
        
        assert_eq!(batch_from_producer.data, BatchData::F64(expected_data));
    }

    #[test]
    fn test_inmemory_producer_partial_last_batch() {
        let config = PipelineConfig {
            batch_size: 5,
            num_qubits: 2,
            encoding_method: "amplitude".to_string(),
            ..Default::default()
        };
        let sample_size = 4; // 2^2
        let data = vec![0.0f64; 16]; // 16 elements = 4 samples
        
        let mut producer = InMemoryProducer {
            data,
            cursor: 0,
            sample_size,
            batch_size: config.batch_size,
            num_qubits: config.num_qubits as usize,
            batches_yielded: 0,
            batch_limit: 10,
        };
        
        // 4 samples total, batch size 5 -> should return 1 batch with 4 samples
        let batch1 = producer.produce().unwrap().unwrap();
        assert_eq!(batch1.batch_n, 4);
        
        let batch2 = producer.produce().unwrap();
        assert!(batch2.is_none());
    }

    #[test]
    fn test_spawn_producer_channel_exhaustion() {
        let config = PipelineConfig {
            total_batches: 3,
            ..Default::default()
        };
        let vector_len = super::vector_len(config.num_qubits, &config.encoding_method);
        let producer = SyntheticProducer::new(config, vector_len);
        
        let (rx, handle) = spawn_producer(producer);
        
        // We expect 3 batches
        assert!(rx.recv().unwrap().is_ok());
        assert!(rx.recv().unwrap().is_ok());
        assert!(rx.recv().unwrap().is_ok());
        
        // Iterator should be exhausted, channel should be closed down successfully
        assert!(rx.recv().is_err());
        handle.join().unwrap();
    }

    #[test]
    fn test_spawn_producer_early_consumer_drop() {
        let config = PipelineConfig {
            total_batches: 1000, // Very large so producer definitely tries to send multiple
            ..Default::default()
        };
        let vector_len = super::vector_len(config.num_qubits, &config.encoding_method);
        let producer = SyntheticProducer::new(config, vector_len);
        
        let (rx, handle) = spawn_producer(producer);
        
        // Let it start
        assert!(rx.recv().unwrap().is_ok());
        
        // Drop rx, closing the channel
        drop(rx);
        
        // Thread should cleanly exit instead of panicking
        handle.join().unwrap();
    }
}
