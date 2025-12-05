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

const BATCH_SIZE: usize = 64;
const VECTOR_LEN: usize = 1024; // 2^10
const NUM_QUBITS: usize = 10;

fn build_sample(seed: u64) -> Vec<f64> {
    (0..VECTOR_LEN)
        .map(|i| {
            let mixed = (i as u64).wrapping_add(seed) % 997;
            (mixed as f64 + 1.0) / 1000.0
        })
        .collect()
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
    let prefetch_depth: usize = env::var("PREFETCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(4);
    let report_interval = Duration::from_secs(1);

    println!("Config:");
    println!("  batch size   : {}", BATCH_SIZE);
    println!("  vector length: {}", VECTOR_LEN);
    println!("  num qubits   : {}", NUM_QUBITS);
    println!("  batches      : {}", total_batches);
    println!("  prefetch     : {}", prefetch_depth);
    println!("  env overrides: BATCHES=<usize> PREFETCH=<usize>");
    println!();

    let (tx, rx) = mpsc::sync_channel(prefetch_depth);

    let producer = thread::spawn(move || {
        for batch_idx in 0..total_batches {
            let mut batch = Vec::with_capacity(BATCH_SIZE);
            let seed_base = (batch_idx * BATCH_SIZE) as u64;
            for i in 0..BATCH_SIZE {
                batch.push(build_sample(seed_base + i as u64));
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
        for sample in batch {
            match engine.encode(&sample, NUM_QUBITS, "amplitude") {
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

        total_vectors += BATCH_SIZE;

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
