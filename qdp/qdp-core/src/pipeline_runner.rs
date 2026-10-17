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

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::path::Path;
use std::time::Instant;

use crate::QdpEngine;
use crate::dlpack::DLManagedTensor;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::Precision;
use crate::io;
use crate::reader::{FloatElem, NullHandling, StreamingDataReader};
use crate::readers::ParquetStreamingReader;
use crate::types::Encoding;

/// Configuration for throughput/latency pipeline runs (Python run_throughput_pipeline_py).
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub device_id: usize,
    pub num_qubits: u32,
    pub batch_size: usize,
    pub total_batches: usize,
    pub encoding: Encoding,
    pub seed: Option<u64>,
    pub warmup_batches: usize,
    pub null_handling: NullHandling,
    /// Pipeline element dtype for synthetic batch fill and `encode_batch` dispatch.
    ///
    /// If [`Encoding::supports_f32`](crate::types::Encoding::supports_f32) is false for the
    /// chosen [`encoding`](PipelineConfig::encoding), [`normalize`](PipelineConfig::normalize)
    /// downgrades this to [`Precision::Float64`] (see `types` module docs: batch f32 is wired
    /// only for encodings with a real `encode_batch_f32` today).
    pub dtype: Precision,
    pub prefetch_depth: usize,
}

impl PipelineConfig {
    /// Normalizes the configuration:
    /// 1. If `dtype` is float32 but the encoding cannot use the f32 batch encode path
    ///    ([`Encoding::supports_f32`](crate::types::Encoding::supports_f32)), falls back to
    ///    float64.
    /// 2. If `prefetch_depth == 0`, auto-computes it from `(num_qubits, batch_size, encoding,
    ///    dtype)` to keep the CPU-side prefetch buffer under ~256 MB.
    pub fn normalize(&mut self) {
        if matches!(self.dtype, Precision::Float32) && !self.encoding.supports_f32() {
            log::info!(
                "float32 pipeline requested but encoding '{}' does not support f32; falling back to f64",
                self.encoding.as_str()
            );
            self.dtype = Precision::Float64;
        }
        if self.prefetch_depth == 0 {
            self.prefetch_depth = compute_optimal_prefetch_depth(
                self.num_qubits as usize,
                self.batch_size,
                self.encoding,
                self.dtype,
            );
            log::debug!(
                "auto prefetch_depth={} (qubits={}, batch={}, encoding={}, dtype={:?})",
                self.prefetch_depth,
                self.num_qubits,
                self.batch_size,
                self.encoding.as_str(),
                self.dtype,
            );
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            num_qubits: 16,
            batch_size: 64,
            total_batches: 100,
            encoding: Encoding::Amplitude,
            seed: None,
            warmup_batches: 0,
            null_handling: NullHandling::FillZero,
            dtype: Precision::Float64,
            prefetch_depth: 0, // 0 = auto-compute in normalize()
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

impl BatchData {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }
}

pub(crate) trait ToBatchData: FloatElem {
    fn wrap(v: Vec<Self>) -> BatchData;
    fn from_recycled(b: BatchData) -> Option<Vec<Self>>;
}

impl ToBatchData for f32 {
    fn wrap(v: Vec<f32>) -> BatchData {
        BatchData::F32(v)
    }
    fn from_recycled(b: BatchData) -> Option<Vec<f32>> {
        match b {
            BatchData::F32(v) => Some(v),
            _ => None,
        }
    }
}

impl ToBatchData for f64 {
    fn wrap(v: Vec<f64>) -> BatchData {
        BatchData::F64(v)
    }
    fn from_recycled(b: BatchData) -> Option<Vec<f64>> {
        match b {
            BatchData::F64(v) => Some(v),
            _ => None,
        }
    }
}

pub struct PrefetchedBatch {
    pub data: BatchData,
    pub batch_n: usize,
    pub sample_size: usize,
    pub num_qubits: usize,
}

pub trait BatchProducer: Send + 'static {
    fn produce(&mut self, recycled: Option<BatchData>) -> Result<Option<PrefetchedBatch>>;
}

/// Compute optimal prefetch depth to keep the CPU-side prefetch buffer under ~256 MB,
/// clamped to [1, 32].
///
/// The CPU prefetch buffer holds `prefetch_depth` batches of raw input data (not
/// encoded state vectors, which live on the GPU). For amplitude encoding the input
/// size dominates; for angle/basis it is tiny, so the cap kicks in at the upper bound.
///
/// | num_qubits | encoding   | bytes/batch (f64, bs=64) | auto depth |
/// |------------|------------ |--------------------------|------------|
/// |  8         | amplitude  |  ~128 KB                 | 32         |
/// | 12         | amplitude  |  ~2 MB                   | 32         |
/// | 16         | amplitude  |  ~32 MB                  | 8          |
/// | 20         | amplitude  |  ~512 MB                 | 1          |
/// |  *         | angle/basis|  tiny                    | 32         |
fn compute_optimal_prefetch_depth(
    num_qubits: usize,
    batch_size: usize,
    encoding: Encoding,
    dtype: Precision,
) -> usize {
    const TARGET_BYTES: usize = 256 * 1024 * 1024; // 256 MB
    const MIN_DEPTH: usize = 1;
    const MAX_DEPTH: usize = 32;

    // Use checked arithmetic throughout; treat any overflow as "extremely large batch"
    // and return MIN_DEPTH so we never buffer more than one batch.
    // `Encoding::vector_len` itself uses `1 << n` which would panic at huge n; defend by
    // recomputing with `checked_shl` for amplitude, which is the only path that can blow up.
    let sample_len: Option<usize> = match encoding {
        Encoding::Amplitude => 1usize.checked_shl(num_qubits as u32),
        _ => Some(encoding.vector_len(num_qubits as u32)),
    };
    let bytes_per_element = dtype.bytes();
    let bytes_per_batch = sample_len
        .and_then(|s| batch_size.checked_mul(s))
        .and_then(|b| b.checked_mul(bytes_per_element));

    match bytes_per_batch {
        None | Some(0) => MIN_DEPTH,
        Some(bpb) => (TARGET_BYTES / bpb).clamp(MIN_DEPTH, MAX_DEPTH),
    }
}

pub struct SyntheticProducer {
    pub config: PipelineConfig,
    pub vector_len: usize,
    pub batch_index: usize,
    pub total_batches: usize,
}

impl SyntheticProducer {
    pub fn new(config: PipelineConfig, vector_len: usize) -> Self {
        let total_batches = config.total_batches;
        Self {
            config,
            vector_len,
            batch_index: 0,
            total_batches,
        }
    }
}

impl BatchProducer for SyntheticProducer {
    fn produce(&mut self, recycled: Option<BatchData>) -> Result<Option<PrefetchedBatch>> {
        if self.batch_index >= self.total_batches {
            return Ok(None);
        }

        let mut data = match recycled {
            Some(BatchData::F32(mut buf)) if matches!(self.config.dtype, Precision::Float32) => {
                buf.resize(self.config.batch_size * self.vector_len, 0.0);
                BatchData::F32(buf)
            }
            Some(BatchData::F64(mut buf)) if matches!(self.config.dtype, Precision::Float64) => {
                buf.resize(self.config.batch_size * self.vector_len, 0.0);
                BatchData::F64(buf)
            }
            _ => {
                if matches!(self.config.dtype, Precision::Float32) {
                    BatchData::F32(vec![0.0f32; self.config.batch_size * self.vector_len])
                } else {
                    BatchData::F64(vec![0.0f64; self.config.batch_size * self.vector_len])
                }
            }
        };

        match &mut data {
            BatchData::F32(buf) => {
                fill_batch_inplace_f32(&self.config, self.batch_index, self.vector_len, buf)
            }
            BatchData::F64(buf) => {
                fill_batch_inplace(&self.config, self.batch_index, self.vector_len, buf)
            }
        }

        self.batch_index += 1;
        Ok(Some(PrefetchedBatch {
            data,
            batch_n: self.config.batch_size,
            sample_size: self.vector_len,
            num_qubits: self.config.num_qubits as usize,
        }))
    }
}

pub struct InMemoryProducer<T: FloatElem = f64> {
    pub data: Vec<T>,
    pub cursor: usize,
    pub sample_size: usize,
    pub batch_size: usize,
    pub num_qubits: usize,
    pub batches_yielded: usize,
    pub batch_limit: usize,
}

impl<T: FloatElem + ToBatchData> BatchProducer for InMemoryProducer<T> {
    fn produce(&mut self, recycled: Option<BatchData>) -> Result<Option<PrefetchedBatch>> {
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
        let slice = &self.data[start..end];

        let data = match recycled.and_then(T::from_recycled) {
            Some(mut buf) => {
                buf.clear();
                buf.extend_from_slice(slice);
                T::wrap(buf)
            }
            None => T::wrap(slice.to_vec()),
        };

        Ok(Some(PrefetchedBatch {
            data,
            batch_n,
            sample_size: self.sample_size,
            num_qubits: self.num_qubits,
        }))
    }
}

pub struct StreamingProducer<T: FloatElem = f64> {
    pub reader: ParquetStreamingReader<T>,
    /// Ring buffer of not-yet-consumed elements. Discarding a consumed prefix is O(1): a
    /// `VecDeque` advances its head instead of shifting the retained tail like `Vec::drain`.
    pub buffer: VecDeque<T>,
    pub read_chunk_scratch: Vec<T>,
    pub sample_size: usize,
    pub batch_size: usize,
    pub num_qubits: usize,
    pub batches_yielded: usize,
    pub batch_limit: usize,
}

impl<T: FloatElem + ToBatchData> BatchProducer for StreamingProducer<T> {
    fn produce(&mut self, recycled: Option<BatchData>) -> Result<Option<PrefetchedBatch>> {
        if self.batches_yielded >= self.batch_limit {
            return Ok(None);
        }
        let required = self.batch_size * self.sample_size;
        while self.buffer.len() < required {
            let written = self.reader.read_chunk(&mut self.read_chunk_scratch)?;
            if written == 0 {
                break;
            }
            // Extend from the slice itself, not `.iter().copied()`: `VecDeque` specializes
            // `Extend<&T> for T: Copy` into a bulk `copy_slice`, while the by-value iterator
            // falls back to element-wise writes.
            self.buffer.extend(&self.read_chunk_scratch[..written]);
        }
        let available_samples = self.buffer.len() / self.sample_size;

        if available_samples == 0 {
            return Ok(None);
        }

        let batch_n = available_samples.min(self.batch_size);
        let take = batch_n * self.sample_size;
        self.batches_yielded += 1;

        let mut buf = match recycled.and_then(T::from_recycled) {
            Some(mut buf) => {
                buf.clear();
                buf
            }
            None => Vec::with_capacity(take),
        };
        // Copy the batch out of the ring's two slices, then drop the drain to discard the prefix.
        // Copying via `as_slices` keeps the batch copy on `extend_from_slice`'s bulk path
        // (`Drain` is not `TrustedLen`, so `extend(drain)` would copy element by element), and
        // the front-anchored `drain(..take)` only advances the head — the retained tail is never
        // shifted. Reusing a recycled Vec keeps the batch copy itself allocation-free.
        let (front, back) = self.buffer.as_slices();
        let n_front = take.min(front.len());
        buf.extend_from_slice(&front[..n_front]);
        if n_front < take {
            buf.extend_from_slice(&back[..take - n_front]);
        }
        drop(self.buffer.drain(..take));
        let data = T::wrap(buf);

        Ok(Some(PrefetchedBatch {
            data,
            batch_n,
            sample_size: self.sample_size,
            num_qubits: self.num_qubits,
        }))
    }
}

type ProducerHandles = (
    std::sync::mpsc::Receiver<Result<PrefetchedBatch>>,
    std::sync::mpsc::Sender<BatchData>,
    std::thread::JoinHandle<()>,
);

fn spawn_producer(
    mut producer: impl BatchProducer,
    prefetch_depth: usize,
) -> Result<ProducerHandles> {
    // If prefetch_depth is 0, default to a minimum of 1 to ensure channel can hold at least 1 item
    let depth = prefetch_depth.max(1);
    let (tx, rx) = std::sync::mpsc::sync_channel(depth);
    let (recycle_tx, recycle_rx) = std::sync::mpsc::channel::<BatchData>();
    let handle = std::thread::Builder::new()
        .name("qdp-prefetch".into())
        .spawn(move || {
            loop {
                let recycled = recycle_rx.try_recv().ok();
                match producer.produce(recycled) {
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
        .map_err(|e| MahoutError::Io(format!("Failed to spawn prefetch thread: {}", e)))?;
    Ok((rx, recycle_tx, handle))
}

/// Default Parquet row group size for streaming reader (tunable).
const DEFAULT_PARQUET_ROW_GROUP_SIZE: usize = 2048;

/// Returns the path extension as lowercase ASCII (e.g. "parquet"), or None if missing/non-UTF8.
fn path_extension_lower(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
}

/// Largest integer exactly representable by an f32 (its mantissa is 24 bits): `2^24`.
/// Basis indices above this would silently change value when narrowed to f32.
const MAX_EXACT_F32_INT: f64 = (1u64 << 24) as f64;

/// Reject an explicit f32 request for a basis file whose indices exceed f32's exact
/// integer range. Basis values are integer state indices, not floats, so narrowing
/// `16_777_217` to `16_777_216` would encode the wrong state. Rather than silently
/// corrupt (or silently widen back to f64), surface the conflict to the caller.
fn reject_basis_f32_out_of_range(data: &BatchData) -> Result<()> {
    if let BatchData::F64(v) = data
        && let Some(&bad) = v.iter().find(|&&x| x > MAX_EXACT_F32_INT)
    {
        return Err(MahoutError::InvalidInput(format!(
            "basis index {bad:.0} exceeds f32's exact integer range ({MAX_EXACT_F32_INT:.0}); \
             narrowing to f32 would encode the wrong state. Use dtype='float64' for this basis file."
        )));
    }
    Ok(())
}

/// f64→f32 narrowing cast: values outside f32 range silently become ±Inf.
fn cast_f64_to_batch_data(
    data: Vec<f64>,
    n: usize,
    s: usize,
    dtype: Precision,
    fmt: &str,
) -> (BatchData, usize, usize) {
    if matches!(dtype, Precision::Float32) {
        log::warn!(
            "{fmt} file loaded as f64, casting to f32: values outside f32 range become ±Inf."
        );
        (
            BatchData::F32(data.iter().map(|&x| x as f32).collect()),
            n,
            s,
        )
    } else {
        (BatchData::F64(data), n, s)
    }
}

fn read_file_by_extension(
    path: &Path,
    null_handling: NullHandling,
    dtype: Precision,
    encoding: Encoding,
) -> Result<(BatchData, usize, usize)> {
    use crate::reader::DataReader;
    // Basis values are integer state indices, not floats; f32's 24-bit mantissa
    // corrupts indices above 2^24. Always read basis as f64, then reject an explicit
    // f32 request whose indices would not survive the narrowing (see below).
    let basis = matches!(encoding, Encoding::Basis);
    let read_dtype = if basis { Precision::Float64 } else { dtype };
    let ext_lower = path_extension_lower(path);
    let ext = ext_lower.as_deref();
    let result = match ext {
        Some("parquet") => {
            if matches!(read_dtype, Precision::Float32) {
                let mut reader =
                    crate::readers::ParquetReader::<f32>::new(path, None, null_handling)?;
                let (data, n, s) = reader.read_batch()?;
                (BatchData::F32(data), n, s)
            } else {
                let mut reader =
                    crate::readers::ParquetReader::<f64>::new(path, None, null_handling)?;
                let (data, n, s) = reader.read_batch()?;
                (BatchData::F64(data), n, s)
            }
        }
        Some("arrow") | Some("feather") | Some("ipc") => {
            let mut reader = crate::readers::ArrowIPCReader::new(path, null_handling)?;
            let (data, n, s) = reader.read_batch()?;
            cast_f64_to_batch_data(data, n, s, read_dtype, "Arrow IPC")
        }
        Some("npy") => {
            let (data, n, s) = io::read_numpy_batch(path)?;
            cast_f64_to_batch_data(data, n, s, read_dtype, "NumPy")
        }
        Some("pt") | Some("pth") => {
            let (data, n, s) = io::read_torch_batch(path)?;
            cast_f64_to_batch_data(data, n, s, read_dtype, "PyTorch")
        }
        Some("pb") => {
            let (data, n, s) = io::read_tensorflow_batch(path)?;
            cast_f64_to_batch_data(data, n, s, read_dtype, "TensorFlow")
        }
        _ => {
            return Err(MahoutError::InvalidInput(format!(
                "Unsupported file extension {:?}. Supported: .parquet, .arrow, .feather, .ipc, .npy, .pt, .pth, .pb",
                path.extension()
            )));
        }
    };
    // basis is always read as f64 above; reject the f32 request only when the indices
    // would actually have been corrupted by the narrowing the caller asked for.
    if basis && matches!(dtype, Precision::Float32) {
        reject_basis_f32_out_of_range(&result.0)?;
    }
    Ok(result)
}

/// Stateful iterator that yields one batch DLPack at a time for Python `for` loop consumption.
/// Reads prefetched batches via a bounded channel.
///
/// # Thread safety
/// `Receiver` is `!Sync`, so `rx` is wrapped in a `Mutex` to satisfy PyO3's `#[pyclass]`
/// `Sync` bound.  In practice, `PyRefMut` guarantees exclusive access, so the lock is
/// never contended at runtime.  `Sender` is already `Send + Sync` and needs no wrapper.
pub struct PipelineIterator {
    engine: QdpEngine,
    config: PipelineConfig,
    rx: std::sync::Mutex<Option<std::sync::mpsc::Receiver<Result<PrefetchedBatch>>>>,
    recycle_tx: Option<std::sync::mpsc::Sender<BatchData>>,
    producer_handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for PipelineIterator {
    fn drop(&mut self) {
        // Drop the recycle sender first to unblock the producer if it is waiting on try_recv.
        drop(self.recycle_tx.take());
        // Close the receiver by taking it out of the Option.  This makes any pending
        // or future tx.send() in the producer thread return Err(SendError), so the
        // producer exits its loop without us having to drain the channel manually.
        // The previous drain-loop approach had a TOCTOU race: after we drained, the
        // producer could refill the sync_channel and block on tx.send() forever
        // while we were waiting on join().
        drop(self.rx.lock().unwrap().take());
        if let Some(handle) = self.producer_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Spawn an in-memory producer over already-loaded `data`. Mirrors
/// [`build_streaming_producer`] so the f32/f64 dispatch in `new_from_file` is a single
/// generic call instead of two byte-for-byte-identical match arms.
fn build_inmemory_producer<T>(
    data: Vec<T>,
    sample_size: usize,
    config: &PipelineConfig,
    batch_limit: usize,
) -> Result<ProducerHandles>
where
    T: FloatElem + ToBatchData,
{
    spawn_producer(
        InMemoryProducer::<T> {
            data,
            cursor: 0,
            sample_size,
            batch_size: config.batch_size,
            num_qubits: config.num_qubits as usize,
            batches_yielded: 0,
            batch_limit,
        },
        config.prefetch_depth,
    )
}

fn build_streaming_producer<T>(
    path: &Path,
    config: &PipelineConfig,
    batch_limit: usize,
) -> Result<ProducerHandles>
where
    T: FloatElem + ToBatchData,
{
    let mut reader = ParquetStreamingReader::<T>::new(
        path,
        Some(DEFAULT_PARQUET_ROW_GROUP_SIZE),
        config.null_handling,
    )?;
    let vector_len = vector_len(config.num_qubits, config.encoding);

    const INITIAL_CHUNK_CAP: usize = 64 * 1024;
    // Buffer must hold at least one complete sample; for amplitude encoding with 17+ qubits
    // vector_len (2^n) exceeds INITIAL_CHUNK_CAP and read_chunk would return 0 on a valid file.
    let initial_cap = INITIAL_CHUNK_CAP.max(vector_len);
    let mut buffer = vec![T::default(); initial_cap];
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
            sample_size,
            vector_len,
            config.num_qubits,
            config.encoding.as_str()
        )));
    }

    buffer.truncate(written);
    let read_chunk_scratch = vec![T::default(); initial_cap];
    // Pre-reserve the streaming steady state so the ring never reallocates mid-run. Each batch
    // drains up to `required` elements from the front while a refill appends up to one
    // `initial_cap` chunk to the back, so live length peaks at `required + initial_cap`.
    // `VecDeque::from` reuses the Vec's existing `initial_cap` capacity (O(1), no realloc), so
    // this only tops it up by `required` — making front-advance realloc-free from batch 0.
    let required = config.batch_size * sample_size;
    let mut buffer = VecDeque::from(buffer);
    buffer.reserve((required + initial_cap).saturating_sub(buffer.len()));
    let producer = StreamingProducer::<T> {
        reader,
        buffer,
        read_chunk_scratch,
        sample_size,
        batch_size: config.batch_size,
        num_qubits: config.num_qubits as usize,
        batches_yielded: 0,
        batch_limit,
    };
    spawn_producer(producer, config.prefetch_depth)
}

impl PipelineIterator {
    pub fn new_synthetic(engine: QdpEngine, mut config: PipelineConfig) -> Result<Self> {
        config.normalize();
        let vector_len = vector_len(config.num_qubits, config.encoding);
        let producer = SyntheticProducer::new(config.clone(), vector_len);
        let prefetch_depth = config.prefetch_depth;
        let (rx, recycle_tx, _producer_handle) = spawn_producer(producer, prefetch_depth)?;
        Ok(Self {
            engine,
            config,
            rx: std::sync::Mutex::new(Some(rx)),
            recycle_tx: Some(recycle_tx),
            producer_handle: Some(_producer_handle),
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
        mut config: PipelineConfig,
        batch_limit: usize,
    ) -> Result<Self> {
        config.normalize();
        let path = path.as_ref();
        let (batch_data, num_samples, sample_size) =
            read_file_by_extension(path, config.null_handling, config.dtype, config.encoding)?;
        let vector_len = vector_len(config.num_qubits, config.encoding);

        // Dimension validation before moving batch_data.
        if sample_size != vector_len {
            return Err(MahoutError::InvalidInput(format!(
                "File feature length {} does not match vector_len {} for num_qubits={}, encoding={}",
                sample_size,
                vector_len,
                config.num_qubits,
                config.encoding.as_str()
            )));
        }
        if batch_data.len() != num_samples * sample_size {
            return Err(MahoutError::InvalidInput(format!(
                "File data length {} is not num_samples ({}) * sample_size ({})",
                batch_data.len(),
                num_samples,
                sample_size
            )));
        }

        let (rx, recycle_tx, _producer_handle) = match batch_data {
            BatchData::F32(data) => {
                build_inmemory_producer::<f32>(data, sample_size, &config, batch_limit)?
            }
            BatchData::F64(data) => {
                build_inmemory_producer::<f64>(data, sample_size, &config, batch_limit)?
            }
        };
        Ok(Self {
            engine,
            config,
            rx: std::sync::Mutex::new(Some(rx)),
            recycle_tx: Some(recycle_tx),
            producer_handle: Some(_producer_handle),
        })
    }

    /// Create a pipeline iterator from a Parquet file using streaming read (Phase 2b).
    /// Only `.parquet` is supported; reduces memory for large files by reading in chunks.
    /// Validates sample_size == vector_len after the first chunk.
    pub fn new_from_file_streaming<P: AsRef<Path>>(
        engine: QdpEngine,
        path: P,
        mut config: PipelineConfig,
        batch_limit: usize,
    ) -> Result<Self> {
        config.normalize();
        let path = path.as_ref();
        if path_extension_lower(path).as_deref() != Some("parquet") {
            return Err(MahoutError::InvalidInput(format!(
                "Streaming file loader supports only .parquet; got extension {:?}. Use .source_file(path) for other formats.",
                path.extension()
            )));
        }

        // Basis values are integer state indices; f32 narrowing corrupts indices above
        // 2^24 (see read_file_by_extension). The streaming reader is chunked so we cannot
        // pre-scan the whole file, so for basis we always stream as f64 (lossless) rather
        // than honoring an f32 request that could silently change states mid-stream.
        //
        // The non-streaming loader can pre-scan and so *rejects* an out-of-range f32 basis
        // request; the chunked streaming path cannot, so it downgrades to f64 instead. Warn
        // so the caller is not left believing the stream ran in f32.
        let f32_requested = matches!(config.dtype, Precision::Float32);
        let is_basis = matches!(config.encoding, Encoding::Basis);
        if f32_requested && is_basis {
            log::warn!(
                "float32 requested for streaming basis file; basis indices are integers and f32 \
                 cannot represent indices above {MAX_EXACT_F32_INT:.0} exactly, so this stream is \
                 read as f64. Use dtype='float64' to silence this warning."
            );
        }
        let use_f32 = f32_requested && !is_basis;
        let (rx, recycle_tx, _producer_handle) = if use_f32 {
            build_streaming_producer::<f32>(path, &config, batch_limit)?
        } else {
            build_streaming_producer::<f64>(path, &config, batch_limit)?
        };
        Ok(Self {
            engine,
            config,
            rx: std::sync::Mutex::new(Some(rx)),
            recycle_tx: Some(recycle_tx),
            producer_handle: Some(_producer_handle),
        })
    }

    /// Returns the next batch as a DLPack pointer; `Ok(None)` when exhausted.
    pub fn next_batch(&mut self) -> Result<Option<*mut DLManagedTensor>> {
        let batch = match self.rx.lock().unwrap().as_ref().unwrap().recv() {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Ok(None),
        };
        let ptr = match &batch.data {
            BatchData::F64(buf) => self.engine.encode_batch_for_pipeline(
                buf,
                batch.batch_n,
                batch.sample_size,
                batch.num_qubits,
                self.config.encoding,
            )?,
            BatchData::F32(buf) => self.engine.encode_batch_f32_for_pipeline(
                buf,
                batch.batch_n,
                batch.sample_size,
                batch.num_qubits,
                self.config.encoding,
            )?,
        };
        if let Some(tx) = &self.recycle_tx {
            let _ = tx.send(batch.data);
        }
        Ok(Some(ptr))
    }
}

/// Vector length per sample for given encoding (used by pipeline and iterator).
pub fn vector_len(num_qubits: u32, encoding: Encoding) -> usize {
    encoding.vector_len(num_qubits)
}

/// Deterministic sample generation matching Python utils.build_sample.
fn fill_sample(seed: u64, out: &mut [f64], encoding: Encoding, num_qubits: usize) -> Result<()> {
    let len = out.len();
    if len == 0 {
        return Ok(());
    }
    match encoding {
        Encoding::Basis => {
            // For basis encoding, use 2^num_qubits as the state space size for mask calculation
            let state_space_size = 1 << num_qubits;
            let mask = (state_space_size - 1) as u64;
            let idx = seed & mask;
            out[0] = idx as f64;
        }
        Encoding::Angle | Encoding::Iqp | Encoding::IqpZ | Encoding::Phase => {
            let scale = (2.0 * PI) / len as f64;
            for (i, v) in out.iter_mut().enumerate() {
                let mixed = (i as u64 + seed) % (len as u64);
                *v = mixed as f64 * scale;
            }
        }
        Encoding::Amplitude => {
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
            config.encoding,
            config.num_qubits as usize,
        );
    }
}

/// Deterministic sample generation for f32.
fn fill_sample_f32(
    seed: u64,
    out: &mut [f32],
    encoding: Encoding,
    num_qubits: usize,
) -> Result<()> {
    let len = out.len();
    if len == 0 {
        return Ok(());
    }
    match encoding {
        Encoding::Basis => {
            let state_space_size = 1 << num_qubits;
            let mask = (state_space_size - 1) as u64;
            let idx = seed & mask;
            out[0] = idx as f32;
        }
        Encoding::Angle | Encoding::Iqp | Encoding::IqpZ | Encoding::Phase => {
            let scale = (2.0 * std::f32::consts::PI) / len as f32;
            for (i, v) in out.iter_mut().enumerate() {
                let mixed = (i as u64 + seed) % (len as u64);
                *v = mixed as f32 * scale;
            }
        }
        Encoding::Amplitude => {
            let mask = (len - 1) as u64;
            let scale = 1.0 / len as f32;
            for (i, v) in out.iter_mut().enumerate() {
                let mixed = (i as u64 + seed) & mask;
                *v = mixed as f32 * scale;
            }
        }
    }
    Ok(())
}

fn fill_batch_inplace_f32(
    config: &PipelineConfig,
    batch_idx: usize,
    vector_len: usize,
    batch_buf: &mut [f32],
) {
    debug_assert_eq!(batch_buf.len(), config.batch_size * vector_len);
    let seed_base = config
        .seed
        .unwrap_or(0)
        .wrapping_add((batch_idx * config.batch_size) as u64);
    for i in 0..config.batch_size {
        let offset = i * vector_len;
        let _ = fill_sample_f32(
            seed_base + i as u64,
            &mut batch_buf[offset..offset + vector_len],
            config.encoding,
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
    let mut config = config.clone();
    config.normalize();

    let engine = QdpEngine::new(config.device_id)?;
    let vector_len = vector_len(config.num_qubits, config.encoding);
    let num_qubits = config.num_qubits as usize;

    // Warmup
    if matches!(config.dtype, Precision::Float32) {
        let mut batch_buf = vec![0.0f32; config.batch_size * vector_len];
        for b in 0..config.warmup_batches {
            fill_batch_inplace_f32(&config, b, vector_len, &mut batch_buf);
            let ptr = engine.encode_batch_f32_for_pipeline(
                &batch_buf,
                config.batch_size,
                vector_len,
                num_qubits,
                config.encoding,
            )?;
            unsafe { release_dlpack(ptr) };
        }
    } else {
        let mut batch_buf = vec![0.0f64; config.batch_size * vector_len];
        for b in 0..config.warmup_batches {
            fill_batch_inplace(&config, b, vector_len, &mut batch_buf);
            let ptr = engine.encode_batch_for_pipeline(
                &batch_buf,
                config.batch_size,
                vector_len,
                num_qubits,
                config.encoding,
            )?;
            unsafe { release_dlpack(ptr) };
        }
    }

    let start = Instant::now();

    let producer = SyntheticProducer::new(config.clone(), vector_len);
    let prefetch_depth = config.prefetch_depth;
    let (rx, recycle_tx, producer_handle) = spawn_producer(producer, prefetch_depth)?;

    // Iteration loop
    let mut total_batches = 0;
    while let Ok(result) = rx.recv() {
        let batch = result?;
        let ptr = match &batch.data {
            BatchData::F64(buf) => engine.encode_batch_for_pipeline(
                buf,
                batch.batch_n,
                batch.sample_size,
                batch.num_qubits,
                config.encoding,
            )?,
            BatchData::F32(buf) => engine.encode_batch_f32_for_pipeline(
                buf,
                batch.batch_n,
                batch.sample_size,
                batch.num_qubits,
                config.encoding,
            )?,
        };
        unsafe { release_dlpack(ptr) };
        total_batches += 1;
        let _ = recycle_tx.send(batch.data);
    }

    let _ = producer_handle.join();

    #[cfg(target_os = "linux")]
    engine.synchronize()?;

    let duration_sec = start.elapsed().as_secs_f64().max(1e-9);
    let total_vectors = total_batches * config.batch_size;
    if total_vectors == 0 {
        return Err(MahoutError::InvalidInput(
            "No vectors processed in pipeline".into(),
        ));
    }
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
            encoding: Encoding::from_str_ci(encoding_method).unwrap(),
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, config.encoding);

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
            encoding: Encoding::from_str_ci(encoding_method).unwrap(),
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, config.encoding);

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
    fn generate_batch_matches_fill_batch_inplace_iqp_z() {
        assert_generate_and_inplace_match("iqp-z");
    }

    #[test]
    fn generate_batch_matches_fill_batch_inplace_iqp() {
        assert_generate_and_inplace_match("iqp");
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
    fn adjacent_batches_differ_iqp_z() {
        assert_adjacent_batches_differ("iqp-z");
    }

    #[test]
    fn adjacent_batches_differ_iqp() {
        assert_adjacent_batches_differ("iqp");
    }

    #[test]
    fn test_seed_none() {
        let config = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding: Encoding::Amplitude,
            seed: None,
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, config.encoding);
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
            encoding: Encoding::Amplitude,
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, config.encoding);
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
            encoding: Encoding::Amplitude,
            seed: Some(123),
            ..Default::default()
        };

        let config_upper = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding: Encoding::from_str_ci("AMPLITUDE").unwrap(),
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config_lower.num_qubits, config_lower.encoding);
        let batch_lower = generate_batch(&config_lower, 0, vector_len);
        let batch_upper = generate_batch(&config_upper, 0, vector_len);
        assert_eq!(batch_lower, batch_upper);
    }

    #[test]
    fn test_amplitude_samples_in_range() {
        let config = PipelineConfig {
            num_qubits: 5,
            batch_size: 8,
            encoding: Encoding::Amplitude,
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, config.encoding);

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
            encoding: Encoding::Amplitude,
            seed: None,
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, config.encoding);
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
            encoding: Encoding::Amplitude,
            seed: Some(123),
            ..Default::default()
        };

        let vector_len = vector_len(config.num_qubits, config.encoding);
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
            encoding: Encoding::Amplitude,
            ..Default::default()
        };
        let vector_len = super::vector_len(config.num_qubits, config.encoding);
        let mut producer = SyntheticProducer::new(config, vector_len);

        let mut count = 0;
        while let Ok(Some(_)) = producer.produce(None) {
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
            encoding: Encoding::Amplitude,
            ..Default::default()
        };
        let vector_len = super::vector_len(config.num_qubits, config.encoding);
        let mut producer = SyntheticProducer::new(config.clone(), vector_len);

        let batch_from_producer = producer.produce(None).unwrap().unwrap();
        let expected_data = generate_batch(&config, 0, vector_len);

        assert_eq!(batch_from_producer.data, BatchData::F64(expected_data));
    }

    #[test]
    fn test_inmemory_producer_partial_last_batch() {
        let config = PipelineConfig {
            batch_size: 5,
            num_qubits: 2,
            encoding: Encoding::Amplitude,
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

        let batch1 = producer.produce(None).unwrap().unwrap();
        assert_eq!(batch1.batch_n, 4);

        let batch2 = producer.produce(None).unwrap();
        assert!(batch2.is_none());
    }

    #[test]
    fn test_spawn_producer_channel_exhaustion() {
        let config = PipelineConfig {
            total_batches: 3,
            prefetch_depth: 16,
            ..Default::default()
        };
        let vector_len = super::vector_len(config.num_qubits, config.encoding);
        let producer = SyntheticProducer::new(config, vector_len);

        let (rx, _recycle_tx, handle) = spawn_producer(producer, 16).unwrap();

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
            prefetch_depth: 16,
            ..Default::default()
        };
        let vector_len = super::vector_len(config.num_qubits, config.encoding);
        let producer = SyntheticProducer::new(config, vector_len);

        let (rx, _recycle_tx, handle) = spawn_producer(producer, 16).unwrap();

        // Let it start
        assert!(rx.recv().unwrap().is_ok());

        // Drop rx, closing the channel
        drop(rx);

        // Thread should cleanly exit instead of panicking
        handle.join().unwrap();
    }

    #[test]
    fn test_synthetic_producer_f32_amplitude() {
        let mut config = PipelineConfig {
            total_batches: 2,
            num_qubits: 3,
            batch_size: 4,
            encoding: Encoding::Amplitude,
            dtype: Precision::Float32,
            ..Default::default()
        };
        config.normalize();
        let vector_len = super::vector_len(config.num_qubits, config.encoding);
        let mut producer = SyntheticProducer::new(config, vector_len);

        let batch = producer.produce(None).unwrap().unwrap();
        assert!(
            matches!(batch.data, BatchData::F32(_)),
            "amplitude with dtype=Float32 should produce F32 data"
        );

        // Verify data is non-zero (was actually filled)
        if let BatchData::F32(ref buf) = batch.data {
            assert!(
                !buf.iter().all(|&v| v == 0.0),
                "batch data should be non-zero"
            );
        }
    }

    #[test]
    fn test_synthetic_producer_f32_for_angle() {
        let mut config = PipelineConfig {
            total_batches: 1,
            num_qubits: 3,
            batch_size: 4,
            encoding: Encoding::Angle,
            dtype: Precision::Float32,
            ..Default::default()
        };
        config.normalize();
        let vector_len = super::vector_len(config.num_qubits, config.encoding);
        let mut producer = SyntheticProducer::new(config, vector_len);

        let batch = producer.produce(None).unwrap().unwrap();
        assert!(
            matches!(batch.data, BatchData::F32(_)),
            "angle with dtype=Float32 should produce F32 batch data"
        );
    }

    #[test]
    fn test_synthetic_producer_f32_for_basis() {
        let mut config = PipelineConfig {
            total_batches: 1,
            num_qubits: 3,
            batch_size: 4,
            encoding: Encoding::Basis,
            dtype: Precision::Float32,
            ..Default::default()
        };
        config.normalize();
        let vector_len = super::vector_len(config.num_qubits, config.encoding);
        let mut producer = SyntheticProducer::new(config, vector_len);

        let batch = producer.produce(None).unwrap().unwrap();
        assert!(
            matches!(batch.data, BatchData::F32(_)),
            "basis with dtype=Float32 should produce F32 batch data"
        );
    }

    #[test]
    fn test_encoding_supports_f32() {
        assert!(Encoding::Amplitude.supports_f32());
        assert!(Encoding::from_str_ci("Amplitude").unwrap().supports_f32());
        assert!(Encoding::from_str_ci("AMPLITUDE").unwrap().supports_f32());
        assert!(Encoding::Angle.supports_f32());
        assert!(Encoding::Basis.supports_f32());
        assert!(!Encoding::Iqp.supports_f32());
        assert!(!Encoding::IqpZ.supports_f32());
        assert!(!Encoding::Phase.supports_f32());
    }

    #[test]
    fn test_vector_len_for_iqp_variants() {
        assert_eq!(super::vector_len(4, Encoding::IqpZ), 4);
        assert_eq!(super::vector_len(4, Encoding::Iqp), 10);
    }

    #[test]
    fn test_iqp_samples_in_angle_range() {
        let config = PipelineConfig {
            num_qubits: 4,
            batch_size: 3,
            encoding: Encoding::Iqp,
            seed: Some(7),
            ..Default::default()
        };

        let vector_len = super::vector_len(config.num_qubits, config.encoding);
        let batch = generate_batch(&config, 0, vector_len);
        let upper = 2.0 * PI;
        for &value in &batch {
            assert!(
                (0.0..upper).contains(&value),
                "iqp value should be in [0, 2pi), got {}",
                value
            );
        }
    }

    #[test]
    fn test_compute_optimal_prefetch_depth_bounds() {
        // Small qubit count → hits MAX_DEPTH cap (32)
        let d =
            super::compute_optimal_prefetch_depth(4, 64, Encoding::Amplitude, Precision::Float64);
        assert_eq!(d, 32, "4 qubits/amplitude should hit max depth");

        // angle/basis are tiny input → should also hit MAX_DEPTH
        let d_angle =
            super::compute_optimal_prefetch_depth(16, 64, Encoding::Angle, Precision::Float64);
        assert_eq!(
            d_angle, 32,
            "angle encoding has small input, should hit max"
        );

        let d_basis =
            super::compute_optimal_prefetch_depth(16, 64, Encoding::Basis, Precision::Float64);
        assert_eq!(
            d_basis, 32,
            "basis encoding has 1-element input, should hit max"
        );

        // Large qubit count amplitude → depth should be ≥ 1 and ≤ 32
        let d_large =
            super::compute_optimal_prefetch_depth(20, 64, Encoding::Amplitude, Precision::Float64);
        assert!(
            (1..=32).contains(&d_large),
            "20 qubits depth out of range: {d_large}"
        );
        // At 20 qubits, amplitude batch is ~512 MB — floor(256M/512M) = 0, clamped to 1
        assert_eq!(d_large, 1);

        // 16 qubits f64, bs=64: 65536*64*8 = 32 MB → 256M/32M = 8
        let d16 =
            super::compute_optimal_prefetch_depth(16, 64, Encoding::Amplitude, Precision::Float64);
        assert_eq!(d16, 8);

        // 16 qubits f32, bs=64: 65536*64*4 = 16 MB → 256M/16M = 16
        let d16_f32 =
            super::compute_optimal_prefetch_depth(16, 64, Encoding::Amplitude, Precision::Float32);
        assert_eq!(d16_f32, 16);
    }

    #[test]
    fn test_normalize_sets_prefetch_depth() {
        // Default config has prefetch_depth=0; normalize() should compute it.
        let mut config = PipelineConfig {
            num_qubits: 16,
            batch_size: 64,
            encoding: Encoding::Amplitude,
            ..Default::default()
        };
        assert_eq!(config.prefetch_depth, 0, "default should be 0 (auto)");
        config.normalize();
        assert!(
            config.prefetch_depth > 0,
            "normalize() must set prefetch_depth > 0"
        );
    }

    // -------------------------------------------------------------------------
    // dtype file-load tests
    //
    // These tests verify that PipelineConfig.dtype is respected when loading
    // from file sources.  They stop at the BatchData variant boundary — the
    // encode kernel (encode_batch_f32_for_pipeline) is CUDA-gated and cannot
    // be exercised in CPU-only CI.  BatchData::F32 is the observable proxy that
    // confirms the f32 kernel would be called on a GPU host; this mirrors the
    // existing convention in test_synthetic_producer_f32_*.
    // -------------------------------------------------------------------------

    mod dtype_file_tests {
        use super::*;
        use arrow::array::{ArrayRef, FixedSizeListBuilder, Float32Builder, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::fs;
        use std::sync::Arc;

        /// Writes `n_samples` f32 samples of 4 features each — matches amplitude encoding with
        /// 2 qubits (2^2=4).
        fn write_f32_parquet_n(path: &std::path::Path, n_samples: usize) {
            let item_field = Arc::new(Field::new("item", DataType::Float32, true));
            let list_field = Field::new("data", DataType::FixedSizeList(item_field, 4), true);
            let schema = Arc::new(Schema::new(vec![list_field]));
            let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 4);
            for _ in 0..n_samples {
                builder.values().append_slice(&[0.25_f32, 0.5, 0.75, 1.0]);
                builder.append(true);
            }
            let array = Arc::new(builder.finish()) as ArrayRef;
            let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            let file = fs::File::create(path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        fn write_f32_parquet(path: &std::path::Path) {
            write_f32_parquet_n(path, 8);
        }

        static FILE_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);

        fn temp_parquet_path(tag: &str) -> std::path::PathBuf {
            let n = FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::env::temp_dir().join(format!(
                "mahout_pipeline_dtype_{tag}_{pid}_{n}.parquet",
                pid = std::process::id(),
            ))
        }

        #[test]
        fn test_read_file_by_extension_f32_parquet_returns_f32_batch_data() {
            let path = temp_parquet_path("f32");
            write_f32_parquet(&path);
            let result = read_file_by_extension(
                &path,
                NullHandling::FillZero,
                Precision::Float32,
                Encoding::Amplitude,
            );
            let _ = fs::remove_file(&path);
            let (batch_data, num_samples, sample_size) = result.unwrap();
            // Assert the actual values, not just the variant: a zeroed/garbled F32 batch
            // would still match `BatchData::F32(_)` but is not what we loaded.
            match batch_data {
                BatchData::F32(buf) => {
                    assert_eq!(buf.len(), 8 * 4);
                    assert!((buf[0] - 0.25).abs() < 1e-6, "first value must round-trip");
                    assert_eq!(&buf[..4], &[0.25_f32, 0.5, 0.75, 1.0]);
                }
                other => {
                    panic!("dtype=Float32 + f32 Parquet must yield BatchData::F32, got {other:?}")
                }
            }
            assert_eq!(num_samples, 8);
            assert_eq!(sample_size, 4);
        }

        #[test]
        fn test_read_file_by_extension_f64_parquet_returns_f64_batch_data() {
            use arrow::array::Float64Builder;
            use arrow::datatypes::DataType;
            let item_field = Arc::new(Field::new("item", DataType::Float64, true));
            let list_field = Field::new("data", DataType::FixedSizeList(item_field, 4), true);
            let schema = Arc::new(Schema::new(vec![list_field]));
            let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), 4);
            for _ in 0..8 {
                builder.values().append_slice(&[0.1_f64, 0.2, 0.3, 0.4]);
                builder.append(true);
            }
            let array = Arc::new(builder.finish()) as ArrayRef;
            let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            let path = temp_parquet_path("f64");
            let file = fs::File::create(&path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();

            let result = read_file_by_extension(
                &path,
                NullHandling::FillZero,
                Precision::Float64,
                Encoding::Amplitude,
            );
            let _ = fs::remove_file(&path);
            let (batch_data, num_samples, sample_size) = result.unwrap();
            match batch_data {
                BatchData::F64(buf) => {
                    assert_eq!(buf.len(), 8 * 4);
                    assert!((buf[0] - 0.1).abs() < 1e-12, "first value must round-trip");
                    assert_eq!(&buf[..4], &[0.1_f64, 0.2, 0.3, 0.4]);
                }
                other => {
                    panic!("dtype=Float64 must yield BatchData::F64 (regression), got {other:?}")
                }
            }
            assert_eq!(num_samples, 8);
            assert_eq!(sample_size, 4);
        }

        /// Basis encoding has sample_size 1 (one integer state index per sample).
        fn write_basis_parquet(path: &std::path::Path, indices: &[f64]) {
            use arrow::array::Float64Builder;
            let item_field = Arc::new(Field::new("item", DataType::Float64, true));
            let list_field = Field::new("data", DataType::FixedSizeList(item_field, 1), true);
            let schema = Arc::new(Schema::new(vec![list_field]));
            let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), 1);
            for &idx in indices {
                builder.values().append_value(idx);
                builder.append(true);
            }
            let array = Arc::new(builder.finish()) as ArrayRef;
            let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            let file = fs::File::create(path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        #[test]
        fn test_basis_f32_request_kept_as_f64_when_indices_fit() {
            // Indices <= 2^24 are exactly representable in f32, so an f32 request is
            // lossless — but basis is always kept as f64 to route through the integer
            // path. Values must round-trip exactly.
            let path = temp_parquet_path("basis_small");
            write_basis_parquet(&path, &[0.0, 1.0, 1024.0, 16_777_216.0]);
            let result = read_file_by_extension(
                &path,
                NullHandling::FillZero,
                Precision::Float32,
                Encoding::Basis,
            );
            let _ = fs::remove_file(&path);
            match result.unwrap().0 {
                BatchData::F64(buf) => {
                    assert_eq!(buf, vec![0.0, 1.0, 1024.0, 16_777_216.0]);
                }
                other => panic!("basis must be kept as F64 to preserve indices, got {other:?}"),
            }
        }

        #[test]
        fn test_basis_f32_request_rejected_when_index_exceeds_f32_range() {
            // 16_777_217 = 2^24 + 1 cannot be represented exactly in f32 (becomes
            // 16_777_216), so an explicit f32 request must be rejected, not silently
            // corrupted. This is the bug from PR #1407 review item #1.
            let path = temp_parquet_path("basis_big_f32");
            write_basis_parquet(&path, &[1.0, 16_777_217.0]);
            let result = read_file_by_extension(
                &path,
                NullHandling::FillZero,
                Precision::Float32,
                Encoding::Basis,
            );
            let _ = fs::remove_file(&path);
            let err = result.expect_err("f32 basis with index > 2^24 must error");
            assert!(
                matches!(err, MahoutError::InvalidInput(_)),
                "expected InvalidInput, got {err:?}"
            );
        }

        #[test]
        fn test_basis_f64_large_index_loads_exactly() {
            // The same large index under an f64 request loads fine and keeps its value.
            let path = temp_parquet_path("basis_big_f64");
            write_basis_parquet(&path, &[1.0, 16_777_217.0]);
            let result = read_file_by_extension(
                &path,
                NullHandling::FillZero,
                Precision::Float64,
                Encoding::Basis,
            );
            let _ = fs::remove_file(&path);
            match result.unwrap().0 {
                BatchData::F64(buf) => assert_eq!(buf, vec![1.0, 16_777_217.0]),
                other => panic!("f64 basis must yield BatchData::F64, got {other:?}"),
            }
        }

        #[test]
        fn test_inmemory_producer_f32_produce_yields_f32_batch_data() {
            let data = vec![0.25_f32, 0.5, 0.75, 1.0, 0.1, 0.2, 0.3, 0.4]; // 2 samples × 4
            let mut producer = InMemoryProducer::<f32> {
                data,
                cursor: 0,
                sample_size: 4,
                batch_size: 2,
                num_qubits: 2,
                batches_yielded: 0,
                batch_limit: 10,
            };
            let batch = producer.produce(None).unwrap().unwrap();
            assert!(
                matches!(batch.data, BatchData::F32(_)),
                "InMemoryProducer::<f32>.produce() must yield BatchData::F32 \
                 so that next_batch() routes to encode_batch_f32_for_pipeline"
            );
            assert_eq!(batch.batch_n, 2);
            assert_eq!(batch.sample_size, 4);
        }

        #[test]
        fn test_streaming_producer_f32_produce_yields_f32_batch_data() {
            let path = temp_parquet_path("streaming_f32");
            write_f32_parquet(&path);

            let result = (|| -> Result<BatchData> {
                let mut reader = ParquetStreamingReader::<f32>::new(
                    &path,
                    Some(DEFAULT_PARQUET_ROW_GROUP_SIZE),
                    NullHandling::FillZero,
                )?;
                const CAP: usize = 64 * 1024;
                let mut buffer = vec![0.0_f32; CAP];
                let written = reader.read_chunk(&mut buffer)?;
                if written == 0 {
                    return Err(MahoutError::InvalidInput("empty file".into()));
                }
                let sample_size = reader.get_sample_size().unwrap();
                buffer.truncate(written);
                let scratch = vec![0.0_f32; CAP];
                let mut producer = StreamingProducer::<f32> {
                    reader,
                    buffer: VecDeque::from(buffer),
                    read_chunk_scratch: scratch,
                    sample_size,
                    batch_size: 4,
                    num_qubits: 2,
                    batches_yielded: 0,
                    batch_limit: 10,
                };
                let batch = producer.produce(None)?.unwrap();
                Ok(batch.data)
            })();

            let _ = fs::remove_file(&path);
            assert!(
                matches!(result.unwrap(), BatchData::F32(_)),
                "StreamingProducer::<f32>.produce() must yield BatchData::F32"
            );
        }

        /// #1436: the `VecDeque` ring buffer must not grow across a long stream. A deliberately
        /// small `read_chunk_scratch` forces `produce()` to refill the ring on nearly every batch,
        /// so the test exercises the real hot-path pattern — front-drain the consumed prefix while
        /// new chunks are appended to the back. A correct ring reuses the freed front slots for the
        /// appended tail, so capacity settles to a constant: front-advance is O(1) amortized with no
        /// reallocation and no per-batch shift of the retained tail. The `refills_observed` assert
        /// guards against a regression where the whole stream is slurped up front (which would make
        /// the capacity check vacuous).
        ///
        /// The same loop also pins down the wrap-around batch copy: once the ring wraps, a batch
        /// can straddle the physical end of the allocation, so `produce()` must stitch it together
        /// from both halves of `as_slices()`. Reaching that case needs a read chunk that is *not*
        /// a multiple of the batch stride: if every batch drained the ring empty, `VecDeque` would
        /// reset the head to 0 on each refill and the ring would never wrap (`split_observed`
        /// asserts the straddling case is actually reached).
        #[test]
        fn test_streaming_producer_buffer_capacity_constant_over_100_batches() {
            const BATCH_SIZE: usize = 4;
            const SAMPLE_LEN: usize = 4; // 2 qubits, amplitude encoding
            const BATCHES: usize = 100;
            // Elements consumed per batch.
            const STRIDE: usize = BATCH_SIZE * SAMPLE_LEN;
            // One sample more than a batch: produce() refills on nearly every batch and always
            // leaves a remainder behind, so the head walks around the ring instead of resetting.
            const READ_CHUNK: usize = STRIDE + SAMPLE_LEN;
            // Each sample written by write_f32_parquet_n, repeated across the batch.
            const SAMPLE: [f32; SAMPLE_LEN] = [0.25, 0.5, 0.75, 1.0];
            let path = temp_parquet_path("streaming_cap");
            // A few extra samples so the 100th produce() is a full batch, not EOF.
            write_f32_parquet_n(&path, BATCH_SIZE * (BATCHES + 5));

            let run = (|| -> Result<()> {
                let reader = ParquetStreamingReader::<f32>::new(
                    &path,
                    Some(DEFAULT_PARQUET_ROW_GROUP_SIZE),
                    NullHandling::FillZero,
                )?;
                // Exactly the steady-state peak the production build path reserves: one batch
                // plus one refill chunk. Sized this tightly, the head wraps within the first few
                // batches, so straddling batch copies are exercised early and often.
                let mut producer = StreamingProducer::<f32> {
                    reader,
                    buffer: VecDeque::with_capacity(STRIDE + READ_CHUNK),
                    read_chunk_scratch: vec![0.0_f32; READ_CHUNK],
                    sample_size: SAMPLE_LEN,
                    batch_size: BATCH_SIZE,
                    num_qubits: 2,
                    batches_yielded: 0,
                    batch_limit: usize::MAX,
                };

                let assert_batch_values = |batch: &PrefetchedBatch, i: usize| match &batch.data {
                    BatchData::F32(v) => {
                        assert_eq!(v.len(), STRIDE, "batch {i} has the wrong length");
                        for (j, x) in v.iter().enumerate() {
                            assert_eq!(
                                *x,
                                SAMPLE[j % SAMPLE_LEN],
                                "batch {i} element {j} corrupted",
                            );
                        }
                    }
                    other => panic!("batch {i} must be F32, got {other:?}"),
                };

                // Capacity after the first produce() is the steady-state baseline.
                let first = producer.produce(None)?.expect("stream must yield a batch");
                assert_batch_values(&first, 0);
                let baseline_cap = producer.buffer.capacity();

                let mut refills_observed = false;
                let mut split_observed = false;
                for i in 1..BATCHES {
                    let len_before = producer.buffer.len();
                    // The ring already holds a full batch that is split across the wrap boundary:
                    // this produce() needs no refill, so it must copy the batch out of both
                    // halves of as_slices().
                    let (front, back) = producer.buffer.as_slices();
                    if len_before >= STRIDE && !back.is_empty() && front.len() < STRIDE {
                        split_observed = true;
                    }

                    let batch = producer.produce(None)?;
                    let batch = batch.unwrap_or_else(|| panic!("batch {i} unexpectedly empty"));
                    assert_batch_values(&batch, i);
                    // A refill ran during this produce() iff the ring gained elements beyond the
                    // one batch (STRIDE) it just drained from the front.
                    if producer.buffer.len() + STRIDE > len_before {
                        refills_observed = true;
                    }
                    assert_eq!(
                        producer.buffer.capacity(),
                        baseline_cap,
                        "buffer capacity grew at batch {i}: front-advance must not reallocate",
                    );
                }
                assert!(
                    refills_observed,
                    "test must exercise back-appends (ring wrap-around), not a single up-front slurp",
                );
                assert!(
                    split_observed,
                    "test must exercise a batch straddling the ring's wrap boundary",
                );
                Ok(())
            })();

            let _ = fs::remove_file(&path);
            run.unwrap();
        }

        /// Direct unit test of the shared f64→f32 narrowing helper used by every
        /// non-Parquet format (Arrow IPC, NumPy, PyTorch, TensorFlow). Covers both the
        /// variant switch and the documented ±Inf behavior for values outside f32 range.
        #[test]
        fn test_cast_f64_to_batch_data_narrows_and_overflows() {
            // f32 request: in-range values cast exactly, out-of-range overflow to +Inf.
            let (batch, n, s) = cast_f64_to_batch_data(
                vec![0.25, 0.5, 1e40, -1e40],
                1,
                4,
                Precision::Float32,
                "test",
            );
            match batch {
                BatchData::F32(buf) => {
                    assert_eq!(buf[0], 0.25_f32);
                    assert_eq!(buf[1], 0.5_f32);
                    assert!(
                        buf[2].is_infinite() && buf[2] > 0.0,
                        "1e40 must overflow to +Inf"
                    );
                    assert!(
                        buf[3].is_infinite() && buf[3] < 0.0,
                        "-1e40 must overflow to -Inf"
                    );
                }
                other => panic!("Float32 request must yield BatchData::F32, got {other:?}"),
            }
            assert_eq!((n, s), (1, 4));

            // f64 request: passthrough, no cast.
            let (batch, _, _) =
                cast_f64_to_batch_data(vec![0.1, 0.2], 1, 2, Precision::Float64, "test");
            assert_eq!(batch, BatchData::F64(vec![0.1, 0.2]));
        }

        /// End-to-end coverage of the Arrow IPC → f32 path through `read_file_by_extension`:
        /// the reader produces f64, the cast helper narrows it to BatchData::F32.
        #[test]
        fn test_read_file_by_extension_arrow_f32_narrows_to_f32_batch_data() {
            use arrow::array::Float64Builder;
            use arrow::ipc::writer::FileWriter as ArrowIpcFileWriter;

            let item_field = Arc::new(Field::new("item", DataType::Float64, true));
            let list_field = Field::new("data", DataType::FixedSizeList(item_field, 4), true);
            let schema = Arc::new(Schema::new(vec![list_field]));
            let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), 4);
            for _ in 0..3 {
                builder.values().append_slice(&[0.25_f64, 0.5, 0.75, 1.0]);
                builder.append(true);
            }
            let array = Arc::new(builder.finish()) as ArrayRef;
            let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();

            let n = FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "mahout_pipeline_dtype_arrow_{pid}_{n}.arrow",
                pid = std::process::id(),
            ));
            {
                let file = fs::File::create(&path).unwrap();
                let mut writer = ArrowIpcFileWriter::try_new(file, &schema).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }

            let result = read_file_by_extension(
                &path,
                NullHandling::FillZero,
                Precision::Float32,
                Encoding::Amplitude,
            );
            let _ = fs::remove_file(&path);
            let (batch_data, num_samples, sample_size) = result.unwrap();
            match batch_data {
                BatchData::F32(buf) => {
                    assert_eq!(buf.len(), 3 * 4);
                    assert_eq!(&buf[..4], &[0.25_f32, 0.5, 0.75, 1.0]);
                }
                other => {
                    panic!("Arrow IPC + dtype=Float32 must narrow to BatchData::F32, got {other:?}")
                }
            }
            assert_eq!(num_samples, 3);
            assert_eq!(sample_size, 4);
        }

        #[test]
        fn test_build_streaming_producer_f32_channel_yields_f32_batch_data() {
            // BatchData::F32 is the observable proxy for the f32 GPU kernel path; the kernel
            // itself is CUDA-gated and cannot be called in CPU-only CI.
            let path = temp_parquet_path("build_sp_f32");
            write_f32_parquet(&path);
            // 2 qubits + Amplitude → vector_len = 2^2 = 4, matching write_f32_parquet's 4 features.
            let config = PipelineConfig {
                num_qubits: 2,
                encoding: Encoding::Amplitude,
                batch_size: 4,
                dtype: Precision::Float32,
                prefetch_depth: 1,
                ..PipelineConfig::default()
            };
            let result = (|| -> Result<BatchData> {
                let (rx, _recycle_tx, _handle) =
                    build_streaming_producer::<f32>(&path, &config, 1)?;
                rx.recv().unwrap().map(|b| b.data)
            })();
            let _ = fs::remove_file(&path);
            assert!(
                matches!(result.unwrap(), BatchData::F32(_)),
                "build_streaming_producer::<f32> must deliver BatchData::F32 through the channel"
            );
        }
    }

    // -------------------------------------------------------------------------
    // #1462 before/after benchmark: `Vec` + cursor vs `VecDeque` ring buffer.
    //
    // `StreamingProducer::produce` is the streaming hot path — every batch copies one batch out
    // of the buffer and discards it. Before #1462 the buffer was a `Vec` plus a read cursor,
    // compacted with `drain(..cursor)` once the cursor passed the half-way mark; each compaction
    // memmoves the entire retained tail, so per-batch cost scales with buffer size. The
    // `VecDeque` advances its head instead, which is O(1) regardless of how much is retained.
    //
    // The benchmark runs both against the real `ParquetStreamingReader`, so the "after" side is
    // the shipped `StreamingProducer` itself rather than a re-creation of it, and the numbers
    // include the I/O the buffer work actually competes with. It is `#[ignore]`d: timings are
    // too machine-dependent to gate CI, and the behavioural guarantee is already covered by
    // `test_streaming_producer_buffer_capacity_constant_over_100_batches`.
    //
    // Override the workload with BENCH_REPEATS=<n> BENCH_BATCHES=<n>:
    //
    //     cargo test -p qdp-core --release --lib buffer_bench -- \
    //         --ignored --nocapture --test-threads=1
    // -------------------------------------------------------------------------
    mod buffer_bench {
        use super::*;
        use arrow::array::{ArrayRef, FixedSizeListBuilder, Float32Builder, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::fs;
        use std::sync::Arc;
        use std::time::Duration;

        /// Pre-#1462 compaction threshold: compact once the cursor reaches `buffer.len() / 2`.
        const LEGACY_BUFFER_COMPACT_DENOM: usize = 2;

        /// Frozen copy of the pre-#1462 `StreamingProducer` — a `Vec` plus a read cursor, with
        /// the consumed prefix drained once the cursor passes the half-way mark. Kept verbatim
        /// (only renamed) so the "before" column measures the code that actually shipped rather
        /// than a paraphrase of it. Do not "fix" this to track `StreamingProducer`: it is the
        /// benchmark's baseline, and changing it invalidates every recorded number.
        struct LegacyStreamingProducer<T: FloatElem = f64> {
            reader: ParquetStreamingReader<T>,
            buffer: Vec<T>,
            buffer_cursor: usize,
            read_chunk_scratch: Vec<T>,
            sample_size: usize,
            batch_size: usize,
            num_qubits: usize,
            batches_yielded: usize,
            batch_limit: usize,
        }

        impl<T: FloatElem + ToBatchData> BatchProducer for LegacyStreamingProducer<T> {
            fn produce(&mut self, recycled: Option<BatchData>) -> Result<Option<PrefetchedBatch>> {
                if self.batches_yielded >= self.batch_limit {
                    return Ok(None);
                }
                let required = self.batch_size * self.sample_size;
                while (self.buffer.len() - self.buffer_cursor) < required {
                    let written = self.reader.read_chunk(&mut self.read_chunk_scratch)?;
                    if written == 0 {
                        break;
                    }
                    self.buffer
                        .extend_from_slice(&self.read_chunk_scratch[..written]);
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

                let data = match recycled.and_then(T::from_recycled) {
                    Some(mut buf) => {
                        buf.clear();
                        buf.extend_from_slice(&self.buffer[start..end]);
                        T::wrap(buf)
                    }
                    None => T::wrap(self.buffer[start..end].to_vec()),
                };

                if self.buffer_cursor >= self.buffer.len() / LEGACY_BUFFER_COMPACT_DENOM {
                    self.buffer.drain(..self.buffer_cursor);
                    self.buffer_cursor = 0;
                }

                Ok(Some(PrefetchedBatch {
                    data,
                    batch_n,
                    sample_size: self.sample_size,
                    num_qubits: self.num_qubits,
                }))
            }
        }

        /// One workload shape. `num_qubits` fixes the sample length under amplitude encoding
        /// (`2^n`), and `read_chunk` is the refill granularity — `build_streaming_producer` uses
        /// `max(64 KiB, vector_len)`, which is what these shapes reproduce.
        struct Shape {
            label: &'static str,
            num_qubits: u32,
            batch_size: usize,
            batches: usize,
        }

        static BENCH_FILE_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);

        fn temp_parquet_path(tag: &str) -> std::path::PathBuf {
            let n = BENCH_FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::env::temp_dir().join(format!(
                "mahout_bench_{tag}_{pid}_{n}.parquet",
                pid = std::process::id(),
            ))
        }

        /// Element `g` of the flattened file, chosen to vary across the stream so the checksum
        /// catches a batch that is correct element-wise but misaligned or reordered.
        fn element(g: usize) -> f32 {
            (g % 4096) as f32 * (1.0 / 4096.0)
        }

        /// Writes `n_samples` samples of `n_features` f32 each, as a FixedSizeList column —
        /// the layout `ParquetStreamingReader` expects.
        fn write_parquet(path: &std::path::Path, n_samples: usize, n_features: usize) {
            let item_field = Arc::new(Field::new("item", DataType::Float32, true));
            let list_field = Field::new(
                "data",
                DataType::FixedSizeList(item_field, n_features as i32),
                true,
            );
            let schema = Arc::new(Schema::new(vec![list_field]));
            let mut builder = FixedSizeListBuilder::new(
                Float32Builder::with_capacity(n_samples * n_features),
                n_features as i32,
            );
            let mut sample = vec![0.0_f32; n_features];
            for s in 0..n_samples {
                for (j, v) in sample.iter_mut().enumerate() {
                    *v = element(s * n_features + j);
                }
                builder.values().append_slice(&sample);
                builder.append(true);
            }
            let array = Arc::new(builder.finish()) as ArrayRef;
            let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            let file = fs::File::create(path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        /// The producer state that both sides share: a reader positioned after its first chunk,
        /// that chunk, the refill scratch buffer, and the sample size the reader detected.
        struct Opened {
            reader: ParquetStreamingReader<f32>,
            first_chunk: Vec<f32>,
            scratch: Vec<f32>,
            sample_size: usize,
        }

        /// Opens the file and reads the first chunk, exactly as `build_streaming_producer` does,
        /// so neither side gets a head start on the other.
        fn open(path: &std::path::Path, read_chunk: usize) -> Result<Opened> {
            let mut reader = ParquetStreamingReader::<f32>::new(
                path,
                Some(DEFAULT_PARQUET_ROW_GROUP_SIZE),
                NullHandling::FillZero,
            )?;
            let mut buffer = vec![0.0_f32; read_chunk];
            let written = reader.read_chunk(&mut buffer)?;
            assert!(written > 0, "benchmark file must not be empty");
            buffer.truncate(written);
            let sample_size = reader
                .get_sample_size()
                .expect("sample_size after first chunk");
            let scratch = vec![0.0_f32; read_chunk];
            Ok(Opened {
                reader,
                first_chunk: buffer,
                scratch,
                sample_size,
            })
        }

        /// Builds the "before" producer the way the pre-#1462 `build_streaming_producer` did.
        fn legacy_producer(
            path: &std::path::Path,
            shape: &Shape,
            read_chunk: usize,
        ) -> Result<LegacyStreamingProducer<f32>> {
            let opened = open(path, read_chunk)?;
            Ok(LegacyStreamingProducer {
                reader: opened.reader,
                buffer: opened.first_chunk,
                buffer_cursor: 0,
                read_chunk_scratch: opened.scratch,
                sample_size: opened.sample_size,
                batch_size: shape.batch_size,
                num_qubits: shape.num_qubits as usize,
                batches_yielded: 0,
                batch_limit: usize::MAX,
            })
        }

        /// Builds the "after" producer the way the current `build_streaming_producer` does,
        /// including the steady-state `reserve` that keeps front-advance realloc-free.
        fn ring_producer(
            path: &std::path::Path,
            shape: &Shape,
            read_chunk: usize,
        ) -> Result<StreamingProducer<f32>> {
            let opened = open(path, read_chunk)?;
            let required = shape.batch_size * opened.sample_size;
            let mut buffer = VecDeque::from(opened.first_chunk);
            buffer.reserve((required + read_chunk).saturating_sub(buffer.len()));
            Ok(StreamingProducer {
                reader: opened.reader,
                buffer,
                read_chunk_scratch: opened.scratch,
                sample_size: opened.sample_size,
                batch_size: shape.batch_size,
                num_qubits: shape.num_qubits as usize,
                batches_yielded: 0,
                batch_limit: usize::MAX,
            })
        }

        /// Runs `batches` batches and returns the time spent inside `produce()` plus an
        /// order-sensitive checksum of everything produced.
        ///
        /// Only the `produce()` calls are timed: the checksum runs between them, so it does not
        /// dilute the measured difference. Each batch's buffer is handed back as the next call's
        /// `recycled` argument, mirroring what `spawn_producer` does — without that, both sides
        /// would allocate a fresh `Vec` per batch and the allocator noise would dominate.
        fn drive(
            producer: &mut impl BatchProducer,
            batches: usize,
        ) -> Result<(Duration, u64, usize)> {
            let mut recycled: Option<BatchData> = None;
            let mut checksum = 0u64;
            let mut elements = 0usize;
            let mut elapsed = Duration::ZERO;

            for i in 0..batches {
                let start = Instant::now();
                let produced = producer.produce(recycled.take())?;
                elapsed += start.elapsed();

                let batch = produced.unwrap_or_else(|| {
                    panic!("stream ran dry at batch {i}: benchmark file is too small")
                });
                match &batch.data {
                    BatchData::F32(v) => {
                        for x in v {
                            checksum = checksum
                                .wrapping_mul(0x100_0000_01b3)
                                .wrapping_add(x.to_bits() as u64);
                        }
                        elements += v.len();
                    }
                    other => panic!("benchmark expects F32 batches, got {other:?}"),
                }
                recycled = Some(batch.data);
            }
            Ok((elapsed, checksum, elements))
        }

        fn env_usize(key: &str, default: usize) -> usize {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v| *v > 0)
                .unwrap_or(default)
        }

        #[test]
        #[ignore = "benchmark; run with `cargo test -p qdp-core --release --lib buffer_bench -- --ignored --nocapture`"]
        fn bench_streaming_buffer_vec_cursor_vs_vecdeque() {
            // Same 64 KiB refill granularity as build_streaming_producer's INITIAL_CHUNK_CAP.
            const READ_CHUNK: usize = 64 * 1024;
            let repeats = env_usize("BENCH_REPEATS", 5);
            let shapes = [
                // Smallest batch relative to the refill chunk (1:16), where the isolated
                // benchmark below shows the legacy compaction paying the most.
                Shape {
                    label: "8 qubits (256 x 16)",
                    num_qubits: 8,
                    batch_size: 16,
                    batches: env_usize("BENCH_BATCHES", 400),
                },
                Shape {
                    label: "8 qubits (256 x 64)",
                    num_qubits: 8,
                    batch_size: 64,
                    batches: env_usize("BENCH_BATCHES", 200),
                },
                Shape {
                    label: "10 qubits (1024 x 64)",
                    num_qubits: 10,
                    batch_size: 64,
                    batches: env_usize("BENCH_BATCHES", 100),
                },
                Shape {
                    label: "12 qubits (4096 x 32)",
                    num_qubits: 12,
                    batch_size: 32,
                    batches: env_usize("BENCH_BATCHES", 60),
                },
            ];

            println!("\nStreamingProducer buffer strategy — {repeats} repeats, best of each");
            println!("read chunk: {READ_CHUNK} elements; env: BENCH_REPEATS, BENCH_BATCHES\n");
            println!(
                "| shape | batches | before (Vec+cursor) | after (VecDeque) | speedup |\n\
                 |---|---|---|---|---|"
            );

            for shape in &shapes {
                let vector_len = vector_len(shape.num_qubits, Encoding::Amplitude);
                let path = temp_parquet_path("bench_buffer");
                // A few spare samples so the last produce() is a full batch, not a short EOF one.
                write_parquet(&path, shape.batch_size * (shape.batches + 2), vector_len);

                let run = (|| -> Result<(Duration, Duration)> {
                    let mut best_legacy = Duration::MAX;
                    let mut best_ring = Duration::MAX;
                    let mut expected: Option<(u64, usize)> = None;

                    // One untimed pass per side warms the page cache and the allocator, so the
                    // first repeat is not systematically slower than the rest.
                    drive(
                        &mut legacy_producer(&path, shape, READ_CHUNK)?,
                        shape.batches,
                    )?;
                    drive(&mut ring_producer(&path, shape, READ_CHUNK)?, shape.batches)?;

                    // Interleaved so any drift in machine state (thermal, cache pressure) lands
                    // on both sides rather than on whichever ran second.
                    for _ in 0..repeats {
                        for (slot, sample) in [
                            (
                                &mut best_legacy,
                                drive(
                                    &mut legacy_producer(&path, shape, READ_CHUNK)?,
                                    shape.batches,
                                )?,
                            ),
                            (
                                &mut best_ring,
                                drive(
                                    &mut ring_producer(&path, shape, READ_CHUNK)?,
                                    shape.batches,
                                )?,
                            ),
                        ] {
                            let (elapsed, checksum, elements) = sample;
                            match expected {
                                None => expected = Some((checksum, elements)),
                                Some(want) => assert_eq!(
                                    (checksum, elements),
                                    want,
                                    "both buffer strategies must produce an identical element \
                                     stream for {}",
                                    shape.label,
                                ),
                            }
                            *slot = (*slot).min(elapsed);
                        }
                    }
                    Ok((best_legacy, best_ring))
                })();

                let _ = fs::remove_file(&path);
                let (legacy, ring) = run.unwrap();
                let per_batch = |d: Duration| d.as_secs_f64() * 1e6 / shape.batches as f64;
                println!(
                    "| {} | {} | {:.2} ms ({:.1} us/batch) | {:.2} ms ({:.1} us/batch) | {:.2}x |",
                    shape.label,
                    shape.batches,
                    legacy.as_secs_f64() * 1e3,
                    per_batch(legacy),
                    ring.as_secs_f64() * 1e3,
                    per_batch(ring),
                    legacy.as_secs_f64() / ring.as_secs_f64(),
                );
            }
            println!();
        }

        // ---------------------------------------------------------------------
        // Isolated buffer cost.
        //
        // The end-to-end bench above answers "what does the pipeline gain", and the answer is
        // "nothing measurable" — Parquet decode is ~99% of produce(). To show what the change
        // actually does, these two models run the *buffer half* of produce() with the reader
        // replaced by a pre-filled scratch slice: refill-append, copy one batch out into a
        // recycled Vec, discard the consumed prefix. They are identical except for the discard,
        // which is the whole of #1462.
        // ---------------------------------------------------------------------

        /// Buffer half of the pre-#1462 `produce()`: consume via a cursor, then compact by
        /// draining the consumed prefix once the cursor passes the half-way mark. The drain
        /// memmoves every retained element.
        fn legacy_buffer_loop(required: usize, chunk: &[f32], batches: usize) -> usize {
            let mut buffer: Vec<f32> = Vec::with_capacity(required + chunk.len());
            let mut cursor = 0usize;
            let mut out: Vec<f32> = Vec::with_capacity(required);
            let mut moved = 0usize;
            for _ in 0..batches {
                while (buffer.len() - cursor) < required {
                    buffer.extend_from_slice(chunk);
                }
                out.clear();
                out.extend_from_slice(&buffer[cursor..cursor + required]);
                cursor += required;
                // `out.len()` is a constant here, so without `black_box` the optimizer is free
                // to delete the copy it is supposed to be measuring.
                moved += std::hint::black_box(&out).len();
                if cursor >= buffer.len() / LEGACY_BUFFER_COMPACT_DENOM {
                    buffer.drain(..cursor);
                    cursor = 0;
                }
            }
            moved
        }

        /// Buffer half of the current `produce()`: copy the batch out of the ring's two slices,
        /// then front-drain it, which only advances the head.
        fn ring_buffer_loop(required: usize, chunk: &[f32], batches: usize) -> usize {
            let mut buffer: VecDeque<f32> = VecDeque::with_capacity(required + chunk.len());
            let mut out: Vec<f32> = Vec::with_capacity(required);
            let mut moved = 0usize;
            for _ in 0..batches {
                while buffer.len() < required {
                    buffer.extend(chunk);
                }
                out.clear();
                let (front, back) = buffer.as_slices();
                let n_front = required.min(front.len());
                out.extend_from_slice(&front[..n_front]);
                if n_front < required {
                    out.extend_from_slice(&back[..required - n_front]);
                }
                drop(buffer.drain(..required));
                // See the matching note in `legacy_buffer_loop`.
                moved += std::hint::black_box(&out).len();
            }
            moved
        }

        #[test]
        #[ignore = "benchmark; run with `cargo test -p qdp-core --release --lib buffer_bench -- --ignored --nocapture`"]
        fn bench_buffer_strategy_isolated() {
            const READ_CHUNK: usize = 64 * 1024;
            let repeats = env_usize("BENCH_REPEATS", 5);
            let batches = env_usize("BENCH_BATCHES", 2000);
            let chunk = vec![1.0_f32; READ_CHUNK];

            // The batch-to-chunk ratio is what drives the difference, not absolute size: legacy
            // compaction memmoves whatever is retained past the cursor, and the smaller a batch
            // is relative to a refill chunk, the more is retained at each compaction. At ratio
            // 1:1 a compaction leaves nothing behind and both strategies do the same work.
            println!("\nBuffer management only, reader replaced by a pre-filled slice");
            println!("{repeats} repeats, best of each; {batches} batches per run");
            println!("refill chunk fixed at {READ_CHUNK} elements\n");
            println!(
                "| batch elements | batch:chunk | before (Vec+cursor) | after (VecDeque) | speedup |\n\
                 |---|---|---|---|---|"
            );

            for required in [4 * 1024_usize, 16 * 1024, 64 * 1024, 256 * 1024] {
                let mut best_legacy = Duration::MAX;
                let mut best_ring = Duration::MAX;
                let mut expected: Option<usize> = None;

                legacy_buffer_loop(required, &chunk, batches);
                ring_buffer_loop(required, &chunk, batches);

                for _ in 0..repeats {
                    for (slot, run) in [
                        (
                            &mut best_legacy,
                            &legacy_buffer_loop as &dyn Fn(usize, &[f32], usize) -> usize,
                        ),
                        (&mut best_ring, &ring_buffer_loop),
                    ] {
                        let start = Instant::now();
                        let moved = run(required, &chunk, batches);
                        let elapsed = start.elapsed();
                        match expected {
                            None => expected = Some(moved),
                            Some(want) => assert_eq!(
                                moved, want,
                                "both strategies must move the same number of elements",
                            ),
                        }
                        *slot = (*slot).min(elapsed);
                    }
                }

                let per_batch = |d: Duration| d.as_secs_f64() * 1e9 / batches as f64;
                let ratio = if required >= READ_CHUNK {
                    format!("{}:1", required / READ_CHUNK)
                } else {
                    format!("1:{}", READ_CHUNK / required)
                };
                println!(
                    "| {} | {} | {:.0} ns/batch | {:.0} ns/batch | {:.2}x |",
                    required,
                    ratio,
                    per_batch(best_legacy),
                    per_batch(best_ring),
                    best_legacy.as_secs_f64() / best_ring.as_secs_f64(),
                );
            }
            println!();
        }
    }
}
