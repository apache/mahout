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

// DLPack protocol for zero-copy GPU memory sharing with PyTorch

#[cfg(test)]
mod dlpack_tests {
    use std::ffi::c_void;

    use cudarc::driver::CudaDevice;
    use qdp_core::dlpack::{synchronize_stream, CUDA_STREAM_LEGACY};
    use qdp_core::gpu::memory::GpuStateVector;

    #[test]
    fn test_dlpack_batch_shape() {
        let device = CudaDevice::new(0).unwrap();

        let num_samples = 4;
        let num_qubits = 2; // 2^2 = 4 elements per sample
        let state_vector = GpuStateVector::new_batch(&device, num_samples, num_qubits)
            .expect("Failed to create batch state vector");

        let dlpack_ptr = state_vector.to_dlpack();
        assert!(!dlpack_ptr.is_null());

        unsafe {
            let tensor = &(*dlpack_ptr).dl_tensor;

            // Verify ndim is 2
            assert_eq!(tensor.ndim, 2, "DLPack tensor should be 2D for batch");

            // Verify shape
            let shape = std::slice::from_raw_parts(tensor.shape, 2);
            assert_eq!(shape[0], num_samples as i64, "Batch size mismatch");
            assert_eq!(shape[1], (1 << num_qubits) as i64, "State size mismatch");

            // Clean up using the deleter
            if let Some(deleter) = (*dlpack_ptr).deleter {
                deleter(dlpack_ptr);
            }
        }
    }

    #[test]
    fn test_dlpack_single_shape() {
        let device = CudaDevice::new(0).unwrap();

        let num_qubits = 2;
        let state_vector =
            GpuStateVector::new(&device, num_qubits).expect("Failed to create state vector");

        let dlpack_ptr = state_vector.to_dlpack();
        assert!(!dlpack_ptr.is_null());

        unsafe {
            let tensor = &(*dlpack_ptr).dl_tensor;

            // Verify ndim is 2 (even for single sample, per the fix)
            assert_eq!(
                tensor.ndim, 2,
                "DLPack tensor should be 2D for single sample"
            );

            // Verify shape
            let shape = std::slice::from_raw_parts(tensor.shape, 2);
            assert_eq!(shape[0], 1, "Batch size should be 1 for single sample");
            assert_eq!(shape[1], (1 << num_qubits) as i64, "State size mismatch");

            // Clean up using the deleter
            if let Some(deleter) = (*dlpack_ptr).deleter {
                deleter(dlpack_ptr);
            }
        }
    }

    /// synchronize_stream(null) is a no-op and returns Ok(()) on all platforms.
    #[test]
    fn test_synchronize_stream_null() {
        unsafe {
            let result = synchronize_stream(std::ptr::null_mut::<c_void>());
            assert!(
                result.is_ok(),
                "synchronize_stream(null) should return Ok(())"
            );
        }
    }

    /// synchronize_stream(CUDA_STREAM_LEGACY) syncs the legacy default stream (Linux + CUDA).
    #[test]
    #[cfg(target_os = "linux")]
    fn test_synchronize_stream_legacy() {
        unsafe {
            let result = synchronize_stream(CUDA_STREAM_LEGACY);
            assert!(
                result.is_ok(),
                "synchronize_stream(CUDA_STREAM_LEGACY) should succeed on Linux with CUDA"
            );
        }
    }
}
