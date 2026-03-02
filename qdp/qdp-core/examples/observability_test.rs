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

// Observability test example
// Tests pool metrics and overlap tracking features
// Run: cargo run -p qdp-core --example observability_test --release

use qdp_core::QdpEngine;
use qdp_core::dlpack::free_dlpack_tensor;
use std::env;

fn main() {
    // Initialize logger - respect RUST_LOG environment variable
    // Don't override the filter level, let RUST_LOG control it
    env_logger::Builder::from_default_env().init();

    println!("=== QDP Observability Test ===");
    println!();

    // Check environment variables
    let enable_pool_metrics = env::var("QDP_ENABLE_POOL_METRICS")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let enable_overlap_tracking = env::var("QDP_ENABLE_OVERLAP_TRACKING")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    println!("Observability Configuration:");
    println!("  QDP_ENABLE_POOL_METRICS: {}", enable_pool_metrics);
    println!("  QDP_ENABLE_OVERLAP_TRACKING: {}", enable_overlap_tracking);
    println!(
        "  RUST_LOG: {}",
        env::var("RUST_LOG").unwrap_or_else(|_| "not set".to_string())
    );
    println!();

    let engine = match QdpEngine::new(0) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("CUDA unavailable or initialization failed: {:?}", e);
            eprintln!();
            eprintln!("Note: Observability features require CUDA to be available.");
            eprintln!("      If CUDA initialization fails, the pipeline will not run");
            eprintln!("      and observability metrics will not be generated.");
            eprintln!();
            eprintln!("To verify the code logic without CUDA, run unit tests:");
            eprintln!("  cargo test -p qdp-core --lib");
            return;
        }
    };

    // Create test data: 18 qubits = 262144 elements (ensures > 1MB threshold for pipeline)
    // Pipeline is only used for data >= 1MB (131072 elements), so we use 18 qubits to ensure
    // we're well above the threshold and will generate multiple chunks for better testing
    const NUM_QUBITS: usize = 18;
    const VECTOR_LEN: usize = 1 << NUM_QUBITS; // 262144 elements = 2MB
    const NUM_SAMPLES: usize = 10;

    println!("Test Configuration:");
    println!("  num qubits: {}", NUM_QUBITS);
    println!("  vector length: {}", VECTOR_LEN);
    println!("  num samples: {}", NUM_SAMPLES);
    println!();

    // Generate test data
    let mut test_data = vec![0.0f64; NUM_SAMPLES * VECTOR_LEN];
    for i in 0..NUM_SAMPLES {
        let offset = i * VECTOR_LEN;
        for j in 0..VECTOR_LEN {
            test_data[offset + j] = (j as f64) / (VECTOR_LEN as f64);
        }
    }

    println!("Running encoding with observability...");
    println!();

    // Use encode() method (not encode_batch) to trigger pipeline and observability
    // encode_batch uses synchronous path and doesn't use the dual-stream pipeline
    // We'll encode each sample individually to trigger the async pipeline
    for i in 0..NUM_SAMPLES {
        let sample = &test_data[i * VECTOR_LEN..(i + 1) * VECTOR_LEN];
        match engine.encode(sample, NUM_QUBITS, "amplitude") {
            Ok(ptr) => {
                if let Err(e) = unsafe { free_dlpack_tensor(ptr) } {
                    eprintln!("✗ Failed to free DLPack tensor for sample {}: {:?}", i, e);
                    return;
                }
            }
            Err(e) => {
                eprintln!("✗ Encoding failed for sample {}: {:?}", i, e);
                return;
            }
        }
    }

    println!("✓ Encoding completed successfully");
    println!();
    println!("Note: If observability is enabled, check the log output above for:");
    if enable_pool_metrics {
        println!("  - Pool Utilization metrics");
    }
    if enable_overlap_tracking {
        println!("  - H2D overlap percentages");
    }

    println!();
    println!("=== Test Complete ===");
}
