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

#[cfg(not(qdp_gpu_platform))]
mod fallback;
#[cfg(qdp_gpu_platform)]
mod linux;
#[cfg(not(any(qdp_gpu_platform, target_os = "windows")))]
mod other;
// Windows non-GPU stub: used only on Windows without the hip feature.
// When qdp_gpu_platform is set (Windows+hip), the linux module is used instead.
#[cfg(all(target_os = "windows", not(qdp_gpu_platform)))]
mod windows;

#[cfg(qdp_gpu_platform)]
pub(crate) use linux::encode_from_parquet;
#[cfg(not(any(qdp_gpu_platform, target_os = "windows")))]
pub(crate) use other::encode_from_parquet;
#[cfg(all(target_os = "windows", not(qdp_gpu_platform)))]
pub(crate) use windows::encode_from_parquet;
