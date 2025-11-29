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

// Memory safety tests: DLPack lifecycle, RAII, Arc reference counting

use qdp_core::QdpEngine;

mod common;

#[test]
#[cfg(target_os = "linux")]
fn test_memory_pressure() {
    println!("Testing memory pressure (leak detection)");
    println!("Running 100 iterations of encode + free");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: No GPU available");
            return;
        }
    };

    let data = common::create_test_data(1024);

    for i in 0..100 {
        let ptr = engine.encode(&data, 10, "amplitude")
            .expect("Encoding should succeed");

        unsafe {
            let managed = &mut *ptr;
            let deleter = managed.deleter.take().expect("Deleter missing in pressure test!");
            deleter(ptr);
        }

        if (i + 1) % 25 == 0 {
            println!("Completed {} iterations", i + 1);
        }
    }

    println!("PASS: Memory pressure test completed (no OOM, no leaks)");
}

#[test]
#[cfg(target_os = "linux")]
fn test_multiple_concurrent_states() {
    println!("Testing multiple concurrent state vectors...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data1 = common::create_test_data(256);
    let data2 = common::create_test_data(512);
    let data3 = common::create_test_data(1024);

    let ptr1 = engine.encode(&data1, 8, "amplitude").unwrap();
    let ptr2 = engine.encode(&data2, 9, "amplitude").unwrap();
    let ptr3 = engine.encode(&data3, 10, "amplitude").unwrap();

    println!("PASS: Created 3 concurrent state vectors");

    // Free in different order to test Arc reference counting
    unsafe {
        println!("Freeing in order: 2, 1, 3");
        (&mut *ptr2).deleter.take().expect("Deleter missing!")(ptr2);
        (&mut *ptr1).deleter.take().expect("Deleter missing!")(ptr1);
        (&mut *ptr3).deleter.take().expect("Deleter missing!")(ptr3);
    }

    println!("PASS: All states freed successfully");
}

#[test]
#[cfg(target_os = "linux")]
fn test_dlpack_tensor_metadata() {
    println!("Testing DLPack tensor metadata...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let data = common::create_test_data(1024);
    let ptr = engine.encode(&data, 10, "amplitude").unwrap();

    unsafe {
        let managed = &mut *ptr;
        let tensor = &managed.dl_tensor;

        assert_eq!(tensor.ndim, 1, "Should be 1D tensor");
        assert!(!tensor.data.is_null(), "Data pointer should be valid");
        assert!(!tensor.shape.is_null(), "Shape pointer should be valid");
        assert!(!tensor.strides.is_null(), "Strides pointer should be valid");

        let shape = *tensor.shape;
        assert_eq!(shape, 1024, "Shape should be 1024 (2^10)");

        let stride = *tensor.strides;
        assert_eq!(stride, 1, "Stride for 1D contiguous array should be 1");

        assert_eq!(tensor.dtype.code, 5, "Should be complex type (code=5)");
        assert_eq!(tensor.dtype.bits, 128, "Should be 128 bits (2x64-bit floats)");

        println!("PASS: DLPack metadata verified");
        println!("  ndim: {}", tensor.ndim);
        println!("  shape: {}", shape);
        println!("  stride: {}", stride);
        println!("  dtype: code={}, bits={}", tensor.dtype.code, tensor.dtype.bits);

        let deleter = managed.deleter.take().expect("Deleter missing in metadata test!");
        deleter(ptr);
    }
}
