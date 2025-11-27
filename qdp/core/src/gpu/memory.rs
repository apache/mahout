use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use kernels::CuDoubleComplex;
use crate::error::{MahoutError, Result};

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
    pub fn new(_device: &CudaDevice, qubits: usize) -> Result<Self> {
        let _size_elements = 1 << qubits;
        
        // Use alloc_zeros for device-side allocation (critical for performance):
        // - No CPU RAM usage (avoids OOM for large states)
        // - No PCIe transfer (GPU hardware zero-fill)
        // - Fast: microseconds vs seconds for 30 qubits (16GB)
        #[cfg(target_os = "linux")]
        {
            // Calls cuMemAlloc + cuMemsetD8 (GPU hardware zero-fill)
            let slice = _device.alloc_zeros::<CuDoubleComplex>(_size_elements)
                .map_err(|e| MahoutError::MemoryAllocation(
                    format!("Failed to allocate {} bytes of GPU memory (qubits={}): {:?}", 
                            _size_elements * std::mem::size_of::<CuDoubleComplex>(), 
                            qubits, 
                            e)
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

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the size in elements (2^n where n is number of qubits)
    pub fn size_elements(&self) -> usize {
        self.size_elements
    }
}
