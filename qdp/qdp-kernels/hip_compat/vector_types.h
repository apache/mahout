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

// HIP forwarding shim for <vector_types.h> (HIP build path only). HIP defines
// double2 / float2 etc. via <hip/hip_runtime.h>, which is already pulled in by
// the cuda_runtime.h shim in this directory, so this header only needs to
// exist for the `#include <vector_types.h>` line to resolve.

#pragma once
#include <hip/hip_runtime.h>
