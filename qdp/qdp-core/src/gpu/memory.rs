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

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use qdp_kernels::CuDoubleComplex;
use crate::error::{MahoutError, Result};

#[cfg(target_os = "linux")]
fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(target_os = "linux")]
fn cuda_error_to_string(code: i32) -> &'static str {
    match code {
        0 => "cudaSuccess",
        2 => "cudaErrorMemoryAllocation",
        3 => "cudaErrorInitializationError",
        30 => "cudaErrorUnknown",
        _ => "Unknown CUDA error",
    }
}

#[cfg(target_os = "linux")]
fn query_cuda_mem_info() -> Result<(usize, usize)> {
    unsafe {
        unsafe extern "C" {
            fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        }

        let mut free_bytes: usize = 0;
        let mut total_bytes: usize = 0;
        let result = cudaMemGetInfo(&mut free_bytes as *mut usize, &mut total_bytes as *mut usize);

        if result != 0 {
            return Err(MahoutError::Cuda(format!(
                "cudaMemGetInfo failed: {} ({})",
                result,
                cuda_error_to_string(result)
            )));
        }

        Ok((free_bytes, total_bytes))
    }
}

#[cfg(target_os = "linux")]
fn build_oom_message(context: &str, requested_bytes: usize, qubits: Option<usize>, free: usize, total: usize) -> String {
    let qubit_hint = qubits
        .map(|q| format!(" (qubits={})", q))
        .unwrap_or_default();

    format!(
        "GPU out of memory during {context}{qubit_hint}: requested {:.2} MiB, free {:.2} MiB / total {:.2} MiB. Reduce qubits or batch size and retry.",
        bytes_to_mib(requested_bytes),
        bytes_to_mib(free),
        bytes_to_mib(total),
    )
}

/// Guard that checks available GPU memory before attempting a large allocation.
///
/// Returns a MemoryAllocation error with a helpful message when the request
/// exceeds the currently reported free memory.
#[cfg(target_os = "linux")]
pub(crate) fn ensure_device_memory_available(requested_bytes: usize, context: &str, qubits: Option<usize>) -> Result<()> {
    let (free, total) = query_cuda_mem_info()?;

    if (requested_bytes as u64) > (free as u64) {
        return Err(MahoutError::MemoryAllocation(build_oom_message(
            context,
            requested_bytes,
            qubits,
            free,
            total,
        )));
    }

    Ok(())
}

/// Wraps CUDA allocation errors with an OOM-aware MahoutError.
#[cfg(target_os = "linux")]
pub(crate) fn map_allocation_error(
    requested_bytes: usize,
    context: &str,
    qubits: Option<usize>,
    source: impl std::fmt::Debug,
) -> MahoutError {
    match query_cuda_mem_info() {
        Ok((free, total)) => {
            if (requested_bytes as u64) > (free as u64) {
                MahoutError::MemoryAllocation(build_oom_message(
                    context,
                    requested_bytes,
                    qubits,
                    free,
                    total,
                ))
            } else {
                MahoutError::MemoryAllocation(format!(
                    "GPU allocation failed during {context}: requested {:.2} MiB. CUDA error: {:?}",
                    bytes_to_mib(requested_bytes),
                    source,
                ))
            }
        }
        Err(e) => MahoutError::MemoryAllocation(format!(
            "GPU allocation failed during {context}: requested {:.2} MiB. Unable to fetch memory info: {:?}; CUDA error: {:?}",
            bytes_to_mib(requested_bytes),
            e,
            source,
        )),
    }
}

/// RAII wrapper for GPU memory buffer
/// Automatically frees GPU memory when dropped
pub struct GpuBufferRaw {
    pub(crate) slice: CudaSlice<CuDoubleComplex>,
}

impl GpuBufferRaw {
    /// Get raw pointer to GPU memory
    ///
    /// # Safety
    /// Valid only while GpuBufferRaw is alive
    pub fn ptr(&self) -> *mut CuDoubleComplex {
        *self.slice.device_ptr() as *mut CuDoubleComplex
    }
}

/// Quantum state vector on GPU
///
/// Manages complex128 array of size 2^n (n = qubits) in GPU memory.
/// Uses Arc for shared ownership (needed for DLPack/PyTorch integration).
/// Thread-safe: Send + Sync
pub struct GpuStateVector {
    // Use Arc to allow DLPack to share ownership
    pub(crate) buffer: Arc<GpuBufferRaw>,
    pub num_qubits: usize,
    pub size_elements: usize,
}

// Safety: CudaSlice and Arc are both Send + Sync
unsafe impl Send for GpuStateVector {}
unsafe impl Sync for GpuStateVector {}

impl GpuStateVector {
    /// Create GPU state vector for n qubits
    /// Allocates 2^n complex numbers on GPU (freed on drop)
    pub fn new(_device: &Arc<CudaDevice>, qubits: usize) -> Result<Self> {
        let _size_elements: usize = 1usize << qubits;

        #[cfg(target_os = "linux")]
        {
            let requested_bytes = _size_elements
                .checked_mul(std::mem::size_of::<CuDoubleComplex>())
                .ok_or_else(|| MahoutError::MemoryAllocation(
                    format!("Requested GPU allocation size overflow (elements={})", _size_elements)
                ))?;

            // Pre-flight check to gracefully fail before cudaMalloc when OOM is obvious
            ensure_device_memory_available(requested_bytes, "state vector allocation", Some(qubits))?;

            // Use uninitialized allocation to avoid memory bandwidth waste.
            // TODO: Consider using a memory pool for input buffers to avoid repeated
            // cudaMalloc overhead in high-frequency encode() calls.
            let slice = unsafe {
                _device.alloc::<CuDoubleComplex>(_size_elements)
            }.map_err(|e| map_allocation_error(
                requested_bytes,
                "state vector allocation",
                Some(qubits),
                e,
            ))?;

            Ok(Self {
                buffer: Arc::new(GpuBufferRaw { slice }),
                num_qubits: qubits,
                size_elements: _size_elements,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Non-Linux: compiles but GPU unavailable
            Err(MahoutError::Cuda("CUDA is only available on Linux. This build does not support GPU operations.".to_string()))
        }
    }

    /// Get raw GPU pointer for DLPack/FFI
    ///
    /// # Safety
    /// Valid while GpuStateVector or any Arc clone is alive
    pub fn ptr(&self) -> *mut CuDoubleComplex {
        self.buffer.ptr()
    }
}
