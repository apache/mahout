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
use std::sync::Mutex;
use std::time::Instant;

use crate::QdpEngine;
use crate::dlpack::DLManagedTensor;
use crate::error::{MahoutError, Result};
use crate::io;
use crate::reader::StreamingDataReader;
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

/// Data source for the pipeline iterator (Phase 1: Synthetic; Phase 2a: InMemory; Phase 2b: Streaming).
pub enum DataSource {
    Synthetic {
        seed: u64,
        batch_index: usize,
        total_batches: usize,
    },
    /// Phase 2a: full file loaded once; iterator slices by batch_size.
    InMemory {
        data: Vec<f64>,
        cursor: usize,
        num_samples: usize,
        sample_size: usize,
        batches_yielded: usize,
        batch_limit: usize,
    },
    /// Phase 2b: stream from Parquet in chunks; iterator refills buffer and encodes by batch.
    /// Reader is in Mutex so PipelineIterator remains Sync (required by PyO3 pyclass).
    Streaming {
        reader: Mutex<ParquetStreamingReader>,
        buffer: Vec<f64>,
        buffer_cursor: usize,
        read_chunk_scratch: Vec<f64>,
        sample_size: usize,
        batch_limit: usize,
        batches_yielded: usize,
    },
}

impl std::fmt::Debug for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSource::Synthetic {
                seed,
                batch_index,
                total_batches,
            } => f
                .debug_struct("Synthetic")
                .field("seed", seed)
                .field("batch_index", batch_index)
                .field("total_batches", total_batches)
                .finish(),
            DataSource::InMemory {
                cursor,
                num_samples,
                sample_size,
                batches_yielded,
                batch_limit,
                ..
            } => f
                .debug_struct("InMemory")
                .field("cursor", cursor)
                .field("num_samples", num_samples)
                .field("sample_size", sample_size)
                .field("batches_yielded", batches_yielded)
                .field("batch_limit", batch_limit)
                .finish(),
            DataSource::Streaming {
                buffer,
                buffer_cursor,
                sample_size,
                batch_limit,
                batches_yielded,
                ..
            } => f
                .debug_struct("Streaming")
                .field("buffer_len", &buffer.len())
                .field("buffer_cursor", buffer_cursor)
                .field("sample_size", sample_size)
                .field("batch_limit", batch_limit)
                .field("batches_yielded", batches_yielded)
                .finish(),
        }
    }
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
fn read_file_by_extension(path: &Path) -> Result<(Vec<f64>, usize, usize)> {
    let ext_lower = path_extension_lower(path);
    let ext = ext_lower.as_deref();
    match ext {
        Some("parquet") => io::read_parquet_batch(path),
        Some("arrow") | Some("feather") | Some("ipc") => io::read_arrow_ipc_batch(path),
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
/// Holds a clone of QdpEngine, PipelineConfig, and source state; reuses generate_batch and encode_batch.
pub struct PipelineIterator {
    engine: QdpEngine,
    config: PipelineConfig,
    source: DataSource,
    vector_len: usize,
}

/// (batch_data, batch_n, sample_size, num_qubits) from one source pull.
type BatchFromSource = (Vec<f64>, usize, usize, usize);

impl PipelineIterator {
    /// Create a new synthetic-data pipeline iterator.
    pub fn new_synthetic(engine: QdpEngine, config: PipelineConfig) -> Result<Self> {
        let vector_len = vector_len(config.num_qubits, &config.encoding_method);
        let source = DataSource::Synthetic {
            seed: config.seed.unwrap_or(0),
            batch_index: 0,
            total_batches: config.total_batches,
        };
        Ok(Self {
            engine,
            config,
            source,
            vector_len,
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
        let (data, num_samples, sample_size) = read_file_by_extension(path)?;
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

        let source = DataSource::InMemory {
            data,
            cursor: 0,
            num_samples,
            sample_size,
            batches_yielded: 0,
            batch_limit,
        };
        Ok(Self {
            engine,
            config,
            source,
            vector_len,
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

        let mut reader = ParquetStreamingReader::new(path, Some(DEFAULT_PARQUET_ROW_GROUP_SIZE))?;
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

        let source = DataSource::Streaming {
            reader: Mutex::new(reader),
            buffer,
            buffer_cursor: 0,
            read_chunk_scratch,
            sample_size,
            batch_limit,
            batches_yielded: 0,
        };
        Ok(Self {
            engine,
            config,
            source,
            vector_len,
        })
    }

    /// Yields the next batch data from the current source; `None` when exhausted.
    /// Returns (batch_data, batch_n, sample_size, num_qubits).
    fn take_batch_from_source(&mut self) -> Result<Option<BatchFromSource>> {
        Ok(match &mut self.source {
            DataSource::Synthetic {
                batch_index,
                total_batches,
                ..
            } => {
                if *batch_index >= *total_batches {
                    None
                } else {
                    let data = generate_batch(&self.config, *batch_index, self.vector_len);
                    *batch_index += 1;
                    Some((
                        data,
                        self.config.batch_size,
                        self.vector_len,
                        self.config.num_qubits as usize,
                    ))
                }
            }
            DataSource::InMemory {
                data,
                cursor,
                sample_size,
                batches_yielded,
                batch_limit,
                ..
            } => {
                if *batches_yielded >= *batch_limit {
                    None
                } else {
                    let remaining = (data.len() - *cursor) / *sample_size;
                    if remaining == 0 {
                        None
                    } else {
                        let batch_n = remaining.min(self.config.batch_size);
                        let start = *cursor;
                        let end = start + batch_n * *sample_size;
                        *cursor = end;
                        *batches_yielded += 1;
                        let slice = data[start..end].to_vec();
                        Some((
                            slice,
                            batch_n,
                            *sample_size,
                            self.config.num_qubits as usize,
                        ))
                    }
                }
            }
            DataSource::Streaming {
                reader,
                buffer,
                buffer_cursor,
                read_chunk_scratch,
                sample_size,
                batch_limit,
                batches_yielded,
            } => {
                if *batches_yielded >= *batch_limit {
                    None
                } else {
                    let required = self.config.batch_size * *sample_size;
                    while (buffer.len() - *buffer_cursor) < required {
                        let r = reader.get_mut().map_err(|e| {
                            MahoutError::Io(format!("Streaming reader mutex poisoned: {}", e))
                        })?;
                        let written = r.read_chunk(read_chunk_scratch)?;
                        if written == 0 {
                            break;
                        }
                        buffer.extend_from_slice(&read_chunk_scratch[..written]);
                    }
                    let available = buffer.len() - *buffer_cursor;
                    let available_samples = available / *sample_size;
                    if available_samples == 0 {
                        None
                    } else {
                        let batch_n = available_samples.min(self.config.batch_size);
                        let start = *buffer_cursor;
                        let end = start + batch_n * *sample_size;
                        *buffer_cursor = end;
                        *batches_yielded += 1;
                        let slice = buffer[start..end].to_vec();
                        if *buffer_cursor >= buffer.len() / BUFFER_COMPACT_DENOM {
                            buffer.drain(..*buffer_cursor);
                            *buffer_cursor = 0;
                        }
                        Some((
                            slice,
                            batch_n,
                            *sample_size,
                            self.config.num_qubits as usize,
                        ))
                    }
                }
            }
        })
    }

    /// Returns the next batch as a DLPack pointer; `Ok(None)` when exhausted.
    pub fn next_batch(&mut self) -> Result<Option<*mut DLManagedTensor>> {
        let Some((batch_data, batch_n, sample_size, num_qubits)) = self.take_batch_from_source()?
        else {
            return Ok(None);
        };
        let ptr = self.engine.encode_batch(
            &batch_data,
            batch_n,
            sample_size,
            num_qubits,
            &self.config.encoding_method,
        )?;
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
fn fill_sample(seed: u64, out: &mut [f64], encoding_method: &str) -> Result<()> {
    let len = out.len();
    if len == 0 {
        return Ok(());
    }
    match encoding_method.to_lowercase().as_str() {
        "basis" => {
            let mask = len.saturating_sub(1) as u64;
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
fn generate_batch(config: &PipelineConfig, batch_idx: usize, vector_len: usize) -> Vec<f64> {
    let seed_base = config
        .seed
        .unwrap_or(0)
        .wrapping_add((batch_idx * config.batch_size) as u64);
    let mut batch = vec![0.0f64; config.batch_size * vector_len];
    for i in 0..config.batch_size {
        let offset = i * vector_len;
        let _ = fill_sample(
            seed_base + i as u64,
            &mut batch[offset..offset + vector_len],
            &config.encoding_method,
        );
    }
    batch
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

    // Warmup
    for b in 0..config.warmup_batches {
        let batch = generate_batch(config, b, vector_len);
        let ptr = engine.encode_batch(
            &batch,
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
    for b in 0..config.total_batches {
        let batch = generate_batch(config, b, vector_len);
        let ptr = engine.encode_batch(
            &batch,
            config.batch_size,
            vector_len,
            num_qubits,
            &config.encoding_method,
        )?;
        unsafe { release_dlpack(ptr) };
    }

    #[cfg(target_os = "linux")]
    engine.synchronize()?;

    let duration_sec = start.elapsed().as_secs_f64().max(1e-9);
    let total_vectors = config.total_batches * config.batch_size;
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
