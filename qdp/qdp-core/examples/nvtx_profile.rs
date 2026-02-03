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

// NVTX profiling example
// Run: cargo run -p qdp-core --example nvtx_profile --features observability --release

use qdp_core::QdpEngine;
use qdp_core::dlpack::free_dlpack_tensor;

fn main() {
    println!("=== NVTX Profiling Example ===");
    println!();

    // Initialize engine
    let engine = match QdpEngine::new(0) {
        Ok(e) => {
            println!("✓ Engine initialized");
            e
        }
        Err(e) => {
            eprintln!("✗ Failed to initialize engine: {:?}", e);
            return;
        }
    };

    // Create test data (large enough to trigger async pipeline)
    let data_len: usize = 262_144; // 2MB of f64, exceeds async threshold
    let data: Vec<f64> = (0..data_len)
        .map(|i| (i as f64) / (data_len as f64))
        .collect();
    println!("✓ Created test data: {} elements", data.len());
    println!();

    println!("Starting encoding (NVTX markers will appear in Nsight Systems)...");
    println!("Expected NVTX markers:");
    println!("  - Mahout::Encode");
    println!("  - CPU::L2Norm");
    println!("  - GPU::Alloc");
    println!("  - GPU::H2DCopy");
    println!("  - GPU::CopyEventRecord");
    println!("  - GPU::H2D_Stage");
    println!("  - GPU::Kernel");
    println!("  - GPU::ComputeSync");
    println!();

    // Perform encoding (this will trigger NVTX markers)
    match engine.encode(&data, 18, "amplitude") {
        Ok(ptr) => {
            println!("✓ Encoding succeeded");
            println!("✓ DLPack pointer: {:p}", ptr);

            // Clean up using shared helper with safety checks
            match unsafe { free_dlpack_tensor(ptr) } {
                Ok(()) => println!("✓ Memory freed"),
                Err(e) => eprintln!("✗ Failed to free DLPack tensor: {:?}", e),
            }
        }
        Err(e) => {
            eprintln!("✗ Encoding failed: {:?}", e);
        }
    }

    println!();
    println!("=== Test Complete ===");
    println!();
    println!("To view NVTX markers, use Nsight Systems:");
    println!(
        "  nsys profile --trace=cuda,nvtx cargo run -p qdp-core --example nvtx_profile --features observability --release"
    );
    println!("Then open the generated .nsys-rep file in Nsight Systems");
}
