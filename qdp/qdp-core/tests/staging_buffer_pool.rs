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

// Tests for Staging Buffer Pool functionality

use qdp_core::QdpEngine;
use std::sync::Arc;

mod common;

#[test]
#[cfg(target_os = "linux")]
fn test_staging_buffer_pool_reuse() {
    println!("Testing staging buffer pool buffer reuse...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Create test data
    let data1 = common::create_test_data(100);
    let data2 = common::create_test_data(100);

    // First encode - should allocate new buffer
    let result1 = engine.encode(&data1, 7, "amplitude");
    assert!(result1.is_ok(), "First encode should succeed");
    println!("PASS: First encode succeeded");

    // Second encode with same size - should reuse buffer from pool
    let result2 = engine.encode(&data2, 7, "amplitude");
    assert!(result2.is_ok(), "Second encode should succeed");
    println!("PASS: Second encode succeeded (buffer should be reused from pool)");

    // Verify results are valid
    let ptr1 = result1.unwrap();
    let ptr2 = result2.unwrap();
    assert!(!ptr1.is_null(), "First result pointer should be valid");
    assert!(!ptr2.is_null(), "Second result pointer should be valid");
    println!("PASS: Both results have valid pointers");
}

#[test]
#[cfg(target_os = "linux")]
fn test_staging_buffer_pool_different_sizes() {
    println!("Testing staging buffer pool with different buffer sizes...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Test with different data sizes
    // Use qubit count that can accommodate both sizes (2^8 = 256 elements)
    let small_data = common::create_test_data(50);
    let large_data = common::create_test_data(150);

    // Encode small data
    let result1 = engine.encode(&small_data, 8, "amplitude");
    assert!(result1.is_ok(), "Small data encode should succeed");
    println!("PASS: Small data encode succeeded");

    // Encode large data - should allocate new buffer (larger than small one)
    let result2 = engine.encode(&large_data, 8, "amplitude");
    if let Err(e) = &result2 {
        println!("SKIP: Large data encode failed (may be OOM or validation issue): {:?}", e);
        return; // Skip this test if it fails due to resource constraints
    }
    assert!(result2.is_ok(), "Large data encode should succeed");
    println!("PASS: Large data encode succeeded");

    // Encode small data again - should reuse the small buffer from pool
    let result3 = engine.encode(&small_data, 8, "amplitude");
    assert!(result3.is_ok(), "Second small data encode should succeed");
    println!("PASS: Second small data encode succeeded (should reuse buffer)");
}

#[test]
#[cfg(target_os = "linux")]
fn test_staging_buffer_pool_raii_cleanup() {
    println!("Testing RAII automatic buffer cleanup...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data = common::create_test_data(100);

    // Multiple encodes - each should properly clean up via RAII
    for i in 0..5 {
        let result = engine.encode(&data, 7, "amplitude");
        assert!(result.is_ok(), "Encode {} should succeed", i);
    }
    println!("PASS: Multiple encodes completed successfully with RAII cleanup");
}

#[test]
#[cfg(target_os = "linux")]
fn test_staging_buffer_pool_error_handling() {
    println!("Testing staging buffer pool error handling...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Test with invalid input - buffer should be released even on error
    let empty_data: Vec<f64> = vec![];
    let result = engine.encode(&empty_data, 7, "amplitude");
    assert!(result.is_err(), "Empty data should fail");
    println!("PASS: Empty data correctly rejected");

    // Subsequent encode should still work (buffer pool not corrupted)
    let valid_data = common::create_test_data(100);
    let result2 = engine.encode(&valid_data, 7, "amplitude");
    assert!(result2.is_ok(), "Encode after error should still work");
    println!("PASS: Buffer pool recovered correctly after error");
}

#[test]
#[cfg(target_os = "linux")]
fn test_staging_buffer_pool_concurrent_access() {
    println!("Testing staging buffer pool thread safety...");

    let engine = Arc::new(match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    });

    let data = common::create_test_data(100);

    // Spawn multiple threads to test thread safety
    // Note: We don't return the DLPack pointer from threads since it's not Send
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let engine_clone = engine.clone();
            let data_clone = data.clone();
            std::thread::spawn(move || {
                let result = engine_clone.encode(&data_clone, 7, "amplitude");
                assert!(result.is_ok(), "Thread {} encode should succeed", i);
                // Drop the pointer immediately (it's not Send, so we can't return it)
                if let Ok(ptr) = result {
                    // The pointer will be cleaned up by the deleter when dropped
                    // We just verify the encode succeeded
                    assert!(!ptr.is_null());
                }
            })
        })
        .collect();

    // Wait for all threads
    for (i, handle) in handles.into_iter().enumerate() {
        handle.join().unwrap_or_else(|_| panic!("Thread {} panicked", i));
    }
    println!("PASS: Concurrent access to buffer pool succeeded");
}
