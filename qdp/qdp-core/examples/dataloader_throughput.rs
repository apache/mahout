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

// DataLoader-style throughput test
// Simulates a QML training loop that keeps the GPU fed with batches of vectors.
// Run: cargo run -p qdp-core --example dataloader_throughput --release

use std::env;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use qdp_core::QdpEngine;

const DEFAULT_BATCH_SIZE: usize = 64;
const DEFAULT_NUM_QUBITS: usize = 10;

fn build_sample(seed: u64, vector_len: usize) -> Vec<f64> {
    // Lightweight deterministic pattern to keep CPU generation cheap
    let mask = (vector_len - 1) as u64; // power-of-two mask instead of modulo
    let scale = 1.0 / vector_len as f64;

    let mut out = Vec::with_capacity(vector_len);
    for i in 0..vector_len {
        let mixed = (i as u64 + seed) & mask;
        out.push(mixed as f64 * scale);
    }
    out
}

fn main() {
    println!("=== QDP DataLoader Throughput ===");

    let engine = match QdpEngine::new(0) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("CUDA unavailable or initialization failed: {:?}", e);
            return;
        }
    };

    let total_batches: usize = env::var("BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let num_qubits: usize = env::var("QUBITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_NUM_QUBITS);
    let vector_len = 1usize << num_qubits;
    let batch_size: usize = env::var("BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_BATCH_SIZE);
    let prefetch_depth: usize = env::var("PREFETCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(16);
    let report_interval = Duration::from_secs(1);

    println!("Config:");
    println!("  batch size   : {}", batch_size);
    println!("  vector length: {}", vector_len);
    println!("  num qubits   : {}", num_qubits);
    println!("  batches      : {}", total_batches);
    println!("  prefetch     : {}", prefetch_depth);
    println!("  env overrides: BATCHES=<usize> PREFETCH=<usize> QUBITS=<usize> BATCH_SIZE=<usize>");
    println!();

    let (tx, rx) = mpsc::sync_channel(prefetch_depth);

    let producer = thread::spawn(move || {
        for batch_idx in 0..total_batches {
            let mut batch = Vec::with_capacity(batch_size);
            let seed_base = (batch_idx * batch_size) as u64;
            for i in 0..batch_size {
                batch.push(build_sample(seed_base + i as u64, vector_len));
            }
            if tx.send(batch).is_err() {
                break;
            }
        }
    });

    let mut total_vectors = 0usize;
    let mut last_report = Instant::now();
    let start = Instant::now();

    for (batch_idx, batch) in rx.iter().enumerate() {
        // NOTE: The DataLoader produces host-side batches of size BATCH_SIZE,
        // but we currently submit each sample to the GPU one-by-one.
        // From the GPU's perspective this is effectively "batch size = 1"
        // per encode call; batching is only happening on the host side.
        for sample in batch {
            match engine.encode(&sample, num_qubits, "amplitude") {
                Ok(ptr) => unsafe {
                    let managed = &mut *ptr;
                    if let Some(deleter) = managed.deleter.take() {
                        deleter(ptr);
                    }
                },
                Err(e) => {
                    eprintln!(
                        "Encode failed on batch {} (vector {}): {:?}",
                        batch_idx,
                        total_vectors,
                        e
                    );
                    return;
                }
            }
        }

        total_vectors += batch_size;

        if last_report.elapsed() >= report_interval {
            let elapsed = start.elapsed().as_secs_f64().max(1e-6);
            let throughput = total_vectors as f64 / elapsed;
            println!(
                "Processed {:4} batches / {:6} vectors -> {:8.1} vectors/sec",
                batch_idx + 1,
                total_vectors,
                throughput
            );
            last_report = Instant::now();
        }

        if batch_idx + 1 >= total_batches {
            break;
        }
    }

    let _ = producer.join();

    let duration = start.elapsed();
    let throughput = total_vectors as f64 / duration.as_secs_f64().max(1e-6);
    println!();
    println!(
        "=== Completed {} batches ({} vectors) in {:.2?} -> {:.1} vectors/sec ===",
        total_batches, total_vectors, duration, throughput
    );
}
