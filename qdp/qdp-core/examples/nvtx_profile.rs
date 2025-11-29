// NVTX profiling example
// Run: cargo run -p qdp-core --example nvtx_profile --features observability --release

use qdp_core::QdpEngine;

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
    
    // Create test data
    let data: Vec<f64> = (0..1024).map(|i| (i as f64) / 1024.0).collect();
    println!("✓ Created test data: {} elements", data.len());
    println!();
    
    println!("Starting encoding (NVTX markers will appear in Nsight Systems)...");
    println!("Expected NVTX markers:");
    println!("  - Mahout::Encode");
    println!("  - CPU::L2Norm");
    println!("  - GPU::Alloc");
    println!("  - GPU::H2DCopy");
    println!("  - GPU::Kernel");
    println!();
    
    // Perform encoding (this will trigger NVTX markers)
    match engine.encode(&data, 10, "amplitude") {
        Ok(ptr) => {
            println!("✓ Encoding succeeded");
            println!("✓ DLPack pointer: {:p}", ptr);
            
            // Clean up
            unsafe {
                let managed = &mut *ptr;
                if let Some(deleter) = managed.deleter.take() {
                    deleter(ptr);
                    println!("✓ Memory freed");
                }
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
    println!("  nsys profile --trace=cuda,nvtx cargo run -p qdp-core --example nvtx_profile --features observability --release");
    println!("Then open the generated .nsys-rep file in Nsight Systems");
}

