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
use std::ffi::c_void;
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use qdp_kernels::{CuComplex, CuDoubleComplex};
use crate::error::{MahoutError, Result};

/// Precision of the GPU state vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Float32,
    Float64,
}

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
pub struct GpuBufferRaw<T> {
    pub(crate) slice: CudaSlice<T>,
}

impl<T> GpuBufferRaw<T> {
    /// Get raw pointer to GPU memory
    ///
    /// # Safety
    /// Valid only while GpuBufferRaw is alive
    pub fn ptr(&self) -> *mut T {
        *self.slice.device_ptr() as *mut T
    }
}

/// Storage wrapper for precision-specific GPU buffers
pub enum BufferStorage {
    F32(GpuBufferRaw<CuComplex>),
    F64(GpuBufferRaw<CuDoubleComplex>),
}

impl BufferStorage {
    fn precision(&self) -> Precision {
        match self {
            BufferStorage::F32(_) => Precision::Float32,
            BufferStorage::F64(_) => Precision::Float64,
        }
    }

    fn ptr_void(&self) -> *mut c_void {
        match self {
            BufferStorage::F32(buf) => buf.ptr() as *mut c_void,
            BufferStorage::F64(buf) => buf.ptr() as *mut c_void,
        }
    }

    fn ptr_f64(&self) -> Option<*mut CuDoubleComplex> {
        match self {
            BufferStorage::F64(buf) => Some(buf.ptr()),
            _ => None,
        }
    }
}

/// Quantum state vector on GPU
///
/// Manages complex array of size 2^n (n = qubits) in GPU memory.
/// Uses Arc for shared ownership (needed for DLPack/PyTorch integration).
/// Thread-safe: Send + Sync
#[derive(Clone)]
pub struct GpuStateVector {
    // Use Arc to allow DLPack to share ownership
    pub(crate) buffer: Arc<BufferStorage>,
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
                buffer: Arc::new(BufferStorage::F64(GpuBufferRaw { slice })),
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

    /// Get current precision of the underlying buffer.
    pub fn precision(&self) -> Precision {
        self.buffer.precision()
    }

    /// Get raw GPU pointer for DLPack/FFI
    ///
    /// # Safety
    /// Valid while GpuStateVector or any Arc clone is alive
    pub fn ptr_void(&self) -> *mut c_void {
        self.buffer.ptr_void()
    }

    /// Returns a double-precision pointer if the buffer stores complex128 data.
    pub fn ptr_f64(&self) -> Option<*mut CuDoubleComplex> {
        self.buffer.ptr_f64()
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

            // Pre-flight check
            ensure_device_memory_available(requested_bytes, "batch state vector allocation", Some(qubits))?;

            let slice = unsafe {
                _device.alloc::<CuDoubleComplex>(total_elements)
            }.map_err(|e| map_allocation_error(
                requested_bytes,
                "batch state vector allocation",
                Some(qubits),
                e,
            ))?;

            Ok(Self {
                buffer: Arc::new(BufferStorage::F64(GpuBufferRaw { slice })),
                num_qubits: qubits,
                size_elements: total_elements,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda("CUDA is only available on Linux. This build does not support GPU operations.".to_string()))
        }
    }

    /// Convert the state vector to the requested precision (GPU-side).
    ///
    /// For now only down-conversion from Float64 -> Float32 is supported.
    pub fn to_precision(&self, device: &Arc<CudaDevice>, target: Precision) -> Result<Self> {
        if self.precision() == target {
            return Ok(self.clone());
        }

        match (self.precision(), target) {
            (Precision::Float64, Precision::Float32) => {
                #[cfg(target_os = "linux")]
                {
                    let requested_bytes = self.size_elements
                        .checked_mul(std::mem::size_of::<CuComplex>())
                        .ok_or_else(|| MahoutError::MemoryAllocation(
                            format!("Requested GPU allocation size overflow (elements={})", self.size_elements)
                        ))?;

                    ensure_device_memory_available(requested_bytes, "state vector precision conversion", Some(self.num_qubits))?;

                    let slice = unsafe {
                        device.alloc::<CuComplex>(self.size_elements)
                    }.map_err(|e| map_allocation_error(
                        requested_bytes,
                        "state vector precision conversion",
                        Some(self.num_qubits),
                        e,
                    ))?;

                    let src_ptr = self.ptr_f64().ok_or_else(|| MahoutError::InvalidInput(
                        "Source state vector is not Float64; cannot convert to Float32".to_string()
                    ))?;

                    let ret = unsafe {
                        qdp_kernels::convert_state_to_float(
                            src_ptr as *const CuDoubleComplex,
                            *slice.device_ptr_mut() as *mut CuComplex,
                            self.size_elements,
                            std::ptr::null_mut(),
                        )
                    };

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(
                            format!("Precision conversion kernel failed: {}", ret)
                        ));
                    }

                    device.synchronize()
                        .map_err(|e| MahoutError::Cuda(format!("Failed to sync after precision conversion: {:?}", e)))?;

                    Ok(Self {
                        buffer: Arc::new(BufferStorage::F32(GpuBufferRaw { slice })),
                        num_qubits: self.num_qubits,
                        size_elements: self.size_elements,
                    })
                }

                #[cfg(not(target_os = "linux"))]
                {
                    Err(MahoutError::Cuda("Precision conversion requires CUDA (Linux)".to_string()))
                }
            }
            _ => Err(MahoutError::NotImplemented(
                "Requested precision conversion is not supported".to_string()
            )),
        }
    }
}
