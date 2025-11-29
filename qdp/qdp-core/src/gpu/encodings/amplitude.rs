// Amplitude encoding: direct state injection with L2 normalization

use std::sync::Arc;
use cudarc::driver::CudaDevice;
use rayon::prelude::*;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;
use super::QuantumEncoder;

#[cfg(target_os = "linux")]
use std::ffi::c_void;
#[cfg(target_os = "linux")]
use cudarc::driver::DevicePtr;
#[cfg(target_os = "linux")]
use qdp_kernels::launch_amplitude_encode;

/// Amplitude encoding: data → normalized quantum amplitudes
/// 
/// Steps: L2 norm (CPU) → GPU allocation → CUDA kernel (normalize + pad)
/// Fast: ~50-100x vs circuit-based methods
pub struct AmplitudeEncoder;

impl QuantumEncoder for AmplitudeEncoder {
    fn encode(
        &self,
        _device: &Arc<CudaDevice>,
        host_data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        // Validate qubits (max 30 = 16GB GPU memory)
        if num_qubits == 0 {
            return Err(MahoutError::InvalidInput(
                "Number of qubits must be at least 1".to_string()
            ));
        }
        if num_qubits > 30 {
            return Err(MahoutError::InvalidInput(
                format!("Number of qubits {} exceeds practical limit of 30", num_qubits)
            ));
        }

        // Validate input data
        if host_data.is_empty() {
            return Err(MahoutError::InvalidInput(
                "Input data cannot be empty".to_string()
            ));
        }

        let state_len = 1 << num_qubits;
        if host_data.len() > state_len {
            return Err(MahoutError::InvalidInput(
                format!("Input data length {} exceeds state vector size {}", host_data.len(), state_len)
            ));
        }

        // Calculate L2 norm (parallel on CPU for speed)
        let norm = {
            crate::profile_scope!("CPU::L2Norm");
            let norm_sq: f64 = host_data.par_iter().map(|x| x * x).sum();
            norm_sq.sqrt()
        };
        
        if norm == 0.0 {
            return Err(MahoutError::InvalidInput("Input data has zero norm".to_string()));
        }

        #[cfg(target_os = "linux")]
        {
            // Allocate GPU state vector
            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new(_device, num_qubits)?
            };

            // Copy input data to GPU (synchronous, zero-copy from slice)
            // TODO : Use async CUDA streams for pipeline overlap
            let input_slice = {
                crate::profile_scope!("GPU::H2DCopy");
                _device.htod_sync_copy(host_data)
                    .map_err(|e| MahoutError::MemoryAllocation(format!("Failed to allocate input buffer: {:?}", e)))?
            };

            // Launch CUDA kernel (CPU-side launch only; execution is asynchronous)
            let ret = {
                crate::profile_scope!("GPU::KernelLaunch");
                unsafe {
                    launch_amplitude_encode(
                        *input_slice.device_ptr() as *const f64,
                        state_vector.ptr() as *mut c_void,
                        host_data.len() as i32,
                        state_len as i32,
                        norm,
                        std::ptr::null_mut(), // default stream
                    )
                }
            };

            if ret != 0 {
                let error_msg = format!(
                    "Kernel launch failed with CUDA error code: {} ({})",
                    ret,
                    cuda_error_to_string(ret)
                );
                return Err(MahoutError::KernelLaunch(error_msg));
            }

            // Block until all work on the device is complete
            {
                crate::profile_scope!("GPU::Synchronize");
                _device
                    .synchronize()
                    .map_err(|e| MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e)))?;
            }

            Ok(state_vector)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda("CUDA unavailable (non-Linux)".to_string()))
        }
    }

    fn name(&self) -> &'static str {
        "amplitude"
    }

    fn description(&self) -> &'static str {
        "Amplitude encoding with L2 normalization"
    }
}

/// Convert CUDA error code to human-readable string
#[cfg(target_os = "linux")]
fn cuda_error_to_string(code: i32) -> &'static str {
    match code {
        0 => "cudaSuccess",
        1 => "cudaErrorInvalidValue",
        2 => "cudaErrorMemoryAllocation",
        3 => "cudaErrorInitializationError",
        4 => "cudaErrorLaunchFailure",
        6 => "cudaErrorInvalidDevice",
        8 => "cudaErrorInvalidConfiguration",
        11 => "cudaErrorInvalidHostPointer",
        12 => "cudaErrorInvalidDevicePointer",
        17 => "cudaErrorInvalidMemcpyDirection",
        30 => "cudaErrorUnknown",
        _ => "Unknown CUDA error",
    }
}

