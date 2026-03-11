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

#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use cudarc::driver::CudaDevice;
#[cfg(target_os = "linux")]
use qdp_core::{Precision, QdpEngine};

/// Creates normalized test data (f64)
#[allow(dead_code)] // Used by multiple test modules
pub fn create_test_data(size: usize) -> Vec<f64> {
    (0..size).map(|i| (i as f64) / (size as f64)).collect()
}

/// Creates normalized test data (f32)
#[allow(dead_code)]
pub fn create_test_data_f32(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) / (size as f32)).collect()
}

/// Returns a CUDA device handle, or `None` when CUDA is unavailable for the test environment.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn cuda_device() -> Option<Arc<CudaDevice>> {
    CudaDevice::new(0).ok()
}

/// Returns a QDP engine, or `None` when GPU-backed engine initialization is unavailable.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn qdp_engine() -> Option<QdpEngine> {
    QdpEngine::new(0).ok()
}

/// Returns a QDP engine with the requested precision, or `None` when unavailable.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn qdp_engine_with_precision(precision: Precision) -> Option<QdpEngine> {
    QdpEngine::new_with_precision(0, precision).ok()
}
