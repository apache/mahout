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

// FFI interface for CUDA kernels
// Kernels in .cu files, compiled via build.rs
// Dummy implementations provided for non-CUDA platforms

pub mod registry;
pub use registry::{
    Function, KernelError, LaunchConfig, config, function, kernels_embedded, launch, module_names,
    module_symbols,
};

// Complex number (matches CUDA's cuDoubleComplex)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuDoubleComplex {
    pub x: f64, // Real part
    pub y: f64, // Imaginary part
}

// Implement DeviceRepr for cudarc compatibility
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::DeviceRepr for CuDoubleComplex {}

// Also implement ValidAsZeroBits for alloc_zeros support
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::ValidAsZeroBits for CuDoubleComplex {}

// Complex number (matches CUDA's cuComplex / cuFloatComplex)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CuComplex {
    pub x: f32, // Real part
    pub y: f32, // Imaginary part
}

// Implement DeviceRepr for cudarc compatibility
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::DeviceRepr for CuComplex {}

// Also implement ValidAsZeroBits for alloc_zeros support
#[cfg(target_os = "linux")]
unsafe impl cudarc::driver::ValidAsZeroBits for CuComplex {}
