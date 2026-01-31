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
use std::time::Instant;

use crate::QdpEngine;
use crate::dlpack::DLManagedTensor;
use crate::error::Result;

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

/// Data source for the pipeline iterator (Phase 1: Synthetic only; Phase 2: File).
#[derive(Debug)]
pub enum DataSource {
    Synthetic {
        seed: u64,
        batch_index: usize,
        total_batches: usize,
    },
}

/// Stateful iterator that yields one batch DLPack at a time for Python `for` loop consumption.
/// Holds a clone of QdpEngine, PipelineConfig, and source state; reuses generate_batch and encode_batch.
pub struct PipelineIterator {
    engine: QdpEngine,
    config: PipelineConfig,
    source: DataSource,
    vector_len: usize,
}

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

    /// Returns the next batch as a DLPack pointer; `Ok(None)` when exhausted.
    pub fn next_batch(&mut self) -> Result<Option<*mut DLManagedTensor>> {
        let (batch_data, num_qubits) = match &mut self.source {
            DataSource::Synthetic {
                batch_index,
                total_batches,
                ..
            } => {
                if *batch_index >= *total_batches {
                    return Ok(None);
                }
                let data = generate_batch(&self.config, *batch_index, self.vector_len);
                *batch_index += 1;
                (data, self.config.num_qubits as usize)
            }
        };
        let ptr = self.engine.encode_batch(
            &batch_data,
            self.config.batch_size,
            self.vector_len,
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
