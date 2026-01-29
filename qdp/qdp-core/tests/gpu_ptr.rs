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

//
// Tests for GPU pointer encoding paths in QdpEngine (f64 + f32).
//

#![cfg(target_os = "linux")]

use qdp_core::{MahoutError, QdpEngine};

#[test]
fn test_encode_from_gpu_ptr_rejects_too_large_num_qubits() {
    println!("Testing QdpEngine::encode_from_gpu_ptr rejects too large num_qubits...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: Failed to initialize QdpEngine on CUDA device 0");
            return;
        }
    };

    // We only care about the precondition checks; pointer will not be dereferenced
    // because the function returns early on InvalidInput due to checked_shl.
    let dummy_ptr: *const f64 = std::ptr::null();
    let input_len = 1usize;

    // Choose num_qubits large enough that 1usize << num_qubits would overflow
    // without the checked_shl guard.
    let num_qubits = usize::BITS as usize;

    let result =
        unsafe { engine.encode_from_gpu_ptr(dummy_ptr, input_len, num_qubits, "amplitude") };

    assert!(
        matches!(
            &result,
            Err(MahoutError::InvalidInput(msg))
                if msg.contains("too large to compute state vector size safely")
        ),
        "Expected InvalidInput error for oversized num_qubits, got: {:?}",
        result
    );
}

#[test]
fn test_encode_batch_from_gpu_ptr_rejects_too_large_num_qubits() {
    println!("Testing QdpEngine::encode_batch_from_gpu_ptr rejects too large num_qubits...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: Failed to initialize QdpEngine on CUDA device 0");
            return;
        }
    };

    let dummy_ptr: *const f64 = std::ptr::null();
    let num_samples = 1usize;
    let sample_size = 1usize;

    let num_qubits = usize::BITS as usize;

    let result = unsafe {
        engine.encode_batch_from_gpu_ptr(
            dummy_ptr,
            num_samples,
            sample_size,
            num_qubits,
            "amplitude",
        )
    };

    assert!(
        matches!(
            &result,
            Err(MahoutError::InvalidInput(msg))
                if msg.contains("too large to compute state vector size safely")
        ),
        "Expected InvalidInput error for oversized num_qubits in batch path, got: {:?}",
        result
    );
}

#[test]
fn test_encode_from_gpu_ptr_rejects_empty_input() {
    println!("Testing QdpEngine::encode_from_gpu_ptr rejects empty input...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: Failed to initialize QdpEngine on CUDA device 0");
            return;
        }
    };

    let dummy_ptr: *const f64 = std::ptr::null();

    let result = unsafe { engine.encode_from_gpu_ptr(dummy_ptr, 0, 2, "amplitude") };

    assert!(
        matches!(
            &result,
            Err(MahoutError::InvalidInput(msg)) if msg.contains("cannot be empty")
        ),
        "Expected InvalidInput error for empty f64 input, got: {:?}",
        result
    );
}

#[test]
fn test_encode_from_gpu_ptr_rejects_wrong_encoding_method() {
    println!("Testing QdpEngine::encode_from_gpu_ptr rejects non-amplitude method...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: Failed to initialize QdpEngine on CUDA device 0");
            return;
        }
    };

    let dummy_ptr: *const f64 = std::ptr::null();

    let result = unsafe { engine.encode_from_gpu_ptr(dummy_ptr, 1, 1, "angle") };

    assert!(
        matches!(
            &result,
            Err(MahoutError::NotImplemented(msg))
                if msg.contains("only supports 'amplitude' method")
        ),
        "Expected NotImplemented error for non-amplitude f64 method, got: {:?}",
        result
    );
}

#[test]
fn test_encode_from_gpu_ptr_rejects_input_too_large_for_state() {
    println!("Testing QdpEngine::encode_from_gpu_ptr rejects input larger than state...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: Failed to initialize QdpEngine on CUDA device 0");
            return;
        }
    };

    let dummy_ptr: *const f64 = std::ptr::null();
    let num_qubits = 2;
    let state_len = 1usize << num_qubits; // safe small value

    let result =
        unsafe { engine.encode_from_gpu_ptr(dummy_ptr, state_len + 1, num_qubits, "amplitude") };

    assert!(
        matches!(
            &result,
            Err(MahoutError::InvalidInput(msg)) if msg.contains("exceeds state vector size")
        ),
        "Expected InvalidInput error for input larger than state, got: {:?}",
        result
    );
}

#[test]
fn test_encode_batch_from_gpu_ptr_rejects_zero_num_samples() {
    println!("Testing QdpEngine::encode_batch_from_gpu_ptr rejects zero num_samples...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: Failed to initialize QdpEngine on CUDA device 0");
            return;
        }
    };

    let dummy_ptr: *const f64 = std::ptr::null();

    let result = unsafe { engine.encode_batch_from_gpu_ptr(dummy_ptr, 0, 1, 1, "amplitude") };

    assert!(
        matches!(
            &result,
            Err(MahoutError::InvalidInput(msg)) if msg.contains("Number of samples cannot be zero")
        ),
        "Expected InvalidInput error for zero num_samples, got: {:?}",
        result
    );
}

#[test]
fn test_encode_batch_from_gpu_ptr_rejects_zero_sample_size() {
    println!("Testing QdpEngine::encode_batch_from_gpu_ptr rejects zero sample_size...");

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: Failed to initialize QdpEngine on CUDA device 0");
            return;
        }
    };

    let dummy_ptr: *const f64 = std::ptr::null();

    let result = unsafe { engine.encode_batch_from_gpu_ptr(dummy_ptr, 1, 0, 1, "amplitude") };

    assert!(
        matches!(
            &result,
            Err(MahoutError::InvalidInput(msg)) if msg.contains("Sample size cannot be zero")
        ),
        "Expected InvalidInput error for zero sample_size, got: {:?}",
        result
    );
}

#[test]
fn test_encode_batch_from_gpu_ptr_rejects_sample_size_too_large() {
    println!(
        "Testing QdpEngine::encode_batch_from_gpu_ptr rejects sample_size larger than state..."
    );

    let engine = match QdpEngine::new(0) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP: Failed to initialize QdpEngine on CUDA device 0");
            return;
        }
    };

    let dummy_ptr: *const f64 = std::ptr::null();
    let num_qubits = 2;
    let state_len = 1usize << num_qubits; // safe small value

    let result = unsafe {
        engine.encode_batch_from_gpu_ptr(dummy_ptr, 1, state_len + 1, num_qubits, "amplitude")
    };

    assert!(
        matches!(
            &result,
            Err(MahoutError::InvalidInput(msg)) if msg.contains("exceeds state vector size")
        ),
        "Expected InvalidInput error for sample_size larger than state, got: {:?}",
        result
    );
}
