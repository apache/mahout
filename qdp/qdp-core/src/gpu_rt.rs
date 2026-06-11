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
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>

//! Single import point for the device runtime types, vendor-selected at
//! compile time. `cudarc` has no ROCm backend, so on the `hip` build these
//! names resolve to the HIP shim in `qdp_kernels::device` instead of cudarc.
//! Both expose the same type names and method signatures, so call sites and
//! integration tests use `crate::gpu_rt::{...}` (or `qdp_core::gpu_rt::{...}`)
//! and compile unchanged on either vendor.

pub use qdp_kernels::device::{
    CudaDevice, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice,
    ValidAsZeroBits,
};
