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

//! Reusable pool of pinned host buffers for staging Disk → Host → GPU transfers.
//! Intended for producer/consumer pipelines that need a small, fixed set of
//! page-locked buffers to avoid repeated cudaHostAlloc / cudaFreeHost.

use std::sync::{Arc, Condvar, Mutex};

use crate::error::{MahoutError, Result};
use crate::gpu::memory::PinnedHostBuffer;

/// Handle that automatically returns a buffer to the pool on drop.
#[cfg(target_os = "linux")]
pub struct PinnedBufferHandle {
    buffer: Option<PinnedHostBuffer>,
    pool: Arc<PinnedBufferPool>,
}

#[cfg(target_os = "linux")]
impl std::ops::Deref for PinnedBufferHandle {
    type Target = PinnedHostBuffer;

    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().expect("Buffer already returned to pool")
    }
}

#[cfg(target_os = "linux")]
impl std::ops::DerefMut for PinnedBufferHandle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().expect("Buffer already returned to pool")
    }
}

#[cfg(target_os = "linux")]
impl Drop for PinnedBufferHandle {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            let mut free = self.pool.free.lock().unwrap();
            free.push(buf);
            self.pool.available_cv.notify_one();
        }
    }
}

/// Pool of pinned host buffers sized for a fixed batch shape.
#[cfg(target_os = "linux")]
pub struct PinnedBufferPool {
    free: Mutex<Vec<PinnedHostBuffer>>,
    available_cv: Condvar,
    capacity: usize,
    elements_per_buffer: usize,
}

#[cfg(target_os = "linux")]
impl PinnedBufferPool {
    /// Create a pool with `pool_size` pinned buffers, each sized for `elements_per_buffer` f64 values.
    pub fn new(pool_size: usize, elements_per_buffer: usize) -> Result<Arc<Self>> {
        if pool_size == 0 {
            return Err(MahoutError::InvalidInput(
                "PinnedBufferPool requires at least one buffer".to_string(),
            ));
        }
        if elements_per_buffer == 0 {
            return Err(MahoutError::InvalidInput(
                "PinnedBufferPool buffer size must be greater than zero".to_string(),
            ));
        }

        let mut buffers = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            buffers.push(PinnedHostBuffer::new(elements_per_buffer)?);
        }

        Ok(Arc::new(Self {
            free: Mutex::new(buffers),
            available_cv: Condvar::new(),
            capacity: pool_size,
            elements_per_buffer,
        }))
    }

    /// Acquire a pinned buffer, blocking until one is available.
    pub fn acquire(self: &Arc<Self>) -> PinnedBufferHandle {
        let mut free = self.free.lock().unwrap();
        loop {
            if let Some(buffer) = free.pop() {
                return PinnedBufferHandle {
                    buffer: Some(buffer),
                    pool: Arc::clone(self),
                };
            }
            free = self.available_cv.wait(free).unwrap();
        }
    }

    /// Try to acquire a pinned buffer from the pool.
    ///
    /// Returns `None` if the pool is currently empty; callers can choose to spin/wait
    /// or fall back to synchronous paths.
    pub fn try_acquire(self: &Arc<Self>) -> Option<PinnedBufferHandle> {
        let mut free = self.free.lock().unwrap();
        free.pop().map(|buffer| PinnedBufferHandle {
            buffer: Some(buffer),
            pool: Arc::clone(self),
        })
    }

    /// Number of buffers currently available.
    pub fn available(&self) -> usize {
        self.free.lock().unwrap().len()
    }

    /// Total number of buffers managed by this pool.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Fixed element capacity for each buffer in the pool.
    pub fn elements_per_buffer(&self) -> usize {
        self.elements_per_buffer
    }
}
