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
use std::ffi::c_void;

#[cfg(target_os = "linux")]
fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Wraps CUDA allocation errors.
#[cfg(target_os = "linux")]
pub(crate) fn map_allocation_error(
    requested_bytes: usize,
    context: &str,
    source: impl std::fmt::Debug,
) -> MahoutError {
    MahoutError::MemoryAllocation(format!(
        "GPU allocation failed during {context}: requested {:.2} MiB. CUDA error: {:?}",
        bytes_to_mib(requested_bytes),
        source,
    ))
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

            // Allocate without pre-flight check
            let slice = unsafe {
                _device.alloc::<CuDoubleComplex>(_size_elements)
            }.map_err(|e| map_allocation_error(
                requested_bytes,
                "state vector allocation",
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
            Err(MahoutError::Cuda("CUDA is only available on Linux.".to_string()))
        }
    }

    /// Get raw GPU pointer for DLPack/FFI
    ///
    /// # Safety
    /// Valid while GpuStateVector or any Arc clone is alive
    pub fn ptr(&self) -> *mut CuDoubleComplex {
        self.buffer.ptr()
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the size in elements (2^n where n is number of qubits)
    pub fn size_elements(&self) -> usize {
        self.size_elements
    }

    /// Create GPU state vector for a batch of samples
    /// Allocates num_samples * 2^qubits complex numbers on GPU
    pub fn new_batch(_device: &Arc<CudaDevice>, num_samples: usize, qubits: usize) -> Result<Self> {
        let single_state_size: usize = 1usize << qubits;
        let total_elements = num_samples.checked_mul(single_state_size)
            .ok_or_else(|| MahoutError::MemoryAllocation(
                format!("Batch size overflow: {} samples * {} elements", num_samples, single_state_size)
            ))?;

        #[cfg(target_os = "linux")]
        {
            let requested_bytes = total_elements
                .checked_mul(std::mem::size_of::<CuDoubleComplex>())
                .ok_or_else(|| MahoutError::MemoryAllocation(
                    format!("Requested GPU allocation size overflow (elements={})", total_elements)
                ))?;

            // Allocate without pre-flight check
            let slice = unsafe {
                _device.alloc::<CuDoubleComplex>(total_elements)
            }.map_err(|e| map_allocation_error(
                requested_bytes,
                "batch state vector allocation",
                e,
            ))?;

            Ok(Self {
                buffer: Arc::new(GpuBufferRaw { slice }),
                num_qubits: qubits,
                size_elements: total_elements,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda("CUDA is only available on Linux.".to_string()))
        }
    }
}

// === Pinned Memory Implementation ===

/// Pinned Host Memory Buffer (Page-Locked)
///
/// Enables DMA for H2D copies, doubling bandwidth and reducing CPU usage.
#[cfg(target_os = "linux")]
pub struct PinnedBuffer {
    ptr: *mut f64,
    size_elements: usize,
}

#[cfg(target_os = "linux")]
impl PinnedBuffer {
    /// Allocate pinned memory
    pub fn new(elements: usize) -> Result<Self> {
        unsafe {
            let bytes = elements * std::mem::size_of::<f64>();
            let mut ptr: *mut c_void = std::ptr::null_mut();

            unsafe extern "C" {
                fn cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: u32) -> i32;
            }

            let ret = cudaHostAlloc(&mut ptr, bytes, 0); // cudaHostAllocDefault

            if ret != 0 {
                return Err(MahoutError::MemoryAllocation(
                    format!("cudaHostAlloc failed with error code: {}", ret)
                ));
            }

            Ok(Self {
                ptr: ptr as *mut f64,
                size_elements: elements,
            })
        }
    }

    /// Get mutable slice to write data into
    pub fn as_slice_mut(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_elements) }
    }

    /// Get raw pointer for CUDA memcpy
    pub fn ptr(&self) -> *const f64 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.size_elements
    }

    pub fn is_empty(&self) -> bool {
        self.size_elements == 0
    }
}

#[cfg(target_os = "linux")]
impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe {
            unsafe extern "C" {
                fn cudaFreeHost(ptr: *mut c_void) -> i32;
            }
            let _ = cudaFreeHost(self.ptr as *mut c_void);
        }
    }
}

// Safety: Pinned memory is accessible from any thread
#[cfg(target_os = "linux")]
unsafe impl Send for PinnedBuffer {}

#[cfg(target_os = "linux")]
unsafe impl Sync for PinnedBuffer {}
