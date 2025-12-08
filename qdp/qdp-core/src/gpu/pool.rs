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

//! GPU Memory Pool (Staging Buffer Pool)
//!
//! Provides reuse of temporary device buffers for H2D copies.
//! This eliminates frequent cudaMalloc/cudaFree overhead in streaming pipelines.

use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use crate::error::{MahoutError, Result};

/// Wrapper to store buffer with its size information
struct PooledBuffer {
    buffer: CudaSlice<u8>,
    size_bytes: usize,
}

/// RAII Guard for staging buffer
///
/// Automatically returns the buffer to the pool when dropped.
/// This ensures buffer is always released, even on early returns or panics.
pub struct BufferGuard<'a> {
    pool: &'a StagingBufferPool,
    buffer: Option<CudaSlice<u8>>,
    size_bytes: usize,
}

impl<'a> Deref for BufferGuard<'a> {
    type Target = CudaSlice<u8>;

    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().unwrap()
    }
}

impl<'a> DerefMut for BufferGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().unwrap()
    }
}

impl<'a> Drop for BufferGuard<'a> {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            self.pool.release_internal(buf, self.size_bytes);
        }
    }
}

impl<'a> BufferGuard<'a> {
    /// Get device pointer as usize (for kernel launch)
    pub fn device_ptr(&self) -> usize {
        *self.deref().device_ptr() as usize
    }

    /// Get device pointer as *const u8
    pub fn device_ptr_u8(&self) -> *const u8 {
        self.device_ptr() as *const u8
    }

    /// Get buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

/// Staging Buffer Pool
///
/// Caches temporary memory (Input Buffer) required for Host -> Device copies.
/// Thread-safe via Mutex.
pub struct StagingBufferPool {
    device: Arc<CudaDevice>,
    // Free list: available buffers with sizes (u8 for generic bytes)
    pool: Mutex<VecDeque<PooledBuffer>>,
    max_pool_size: usize,
}

impl StagingBufferPool {
    /// Create a new staging buffer pool
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            pool: Mutex::new(VecDeque::new()),
            max_pool_size: 8, // Keep max 8 buffers to prevent memory bloat
        }
    }

    /// Acquire a buffer with at least `size_bytes` capacity
    ///
    /// Strategy: First-Fit. Returns the first available buffer that fits.
    /// If no suitable buffer is found, allocates a new one.
    ///
    /// Returns a `BufferGuard` that automatically releases the buffer when dropped.
    pub fn acquire(&self, size_bytes: usize) -> Result<BufferGuard<'_>> {
        let mut pool = self.pool.lock().unwrap();

        // Try to reuse existing buffer, otherwise allocate new (min 1MB for reuse)
        let (buffer, actual_size) = if let Some(idx) = pool.iter().position(|pb| pb.size_bytes >= size_bytes) {
            let pooled = pool.remove(idx).unwrap();
            (pooled.buffer, pooled.size_bytes)
        } else {
            let alloc_size = size_bytes.max(1024 * 1024); // Min 1MB

            let buffer = unsafe {
                self.device.alloc::<u8>(alloc_size)
            }.map_err(|e| MahoutError::MemoryAllocation(format!("Staging alloc failed: {:?}", e)))?;

            (buffer, alloc_size)
        };

        Ok(BufferGuard {
            pool: self,
            buffer: Some(buffer),
            size_bytes: actual_size,
        })
    }

    /// Internal method to release buffer (called by BufferGuard::drop)
    fn release_internal(&self, buffer: CudaSlice<u8>, size_bytes: usize) {
        let mut pool = self.pool.lock().unwrap();

        // Keep buffer if pool not full, otherwise drop (triggers cudaFree)
        if pool.len() < self.max_pool_size {
            pool.push_back(PooledBuffer {
                buffer,
                size_bytes,
            });
        }
    }

    /// Manually release a buffer (prefer BufferGuard for automatic cleanup)
    pub fn release(&self, buffer: CudaSlice<u8>, size_bytes: usize) {
        self.release_internal(buffer, size_bytes);
    }

    /// Clear all buffers (freeing memory)
    pub fn clear(&self) {
        let mut pool = self.pool.lock().unwrap();
        pool.clear();
    }
}
