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

// ozaki_config.h — shared configuration types for the Ozaki TC GEMM path.
// Included by ImplicitHadamardOzaki.h only.
#pragma once

#include <cstddef>

namespace ozaki {

struct PreScanStats {
    double dynamic_range;
    double sparsity;
    int    effective_rank;
};

enum class ExecutionMode {
    Phase22,
    Phase23Hetero,
    Phase24ExtremeMix,
    Phase25Stacking,
    Phase26HybridOzaki
};

struct OzakiConfig {
    double        target_epsilon           = 1e-10;
    size_t        vram_limit_mb            = 6144;
    ExecutionMode mode                     = ExecutionMode::Phase22;
};

} // namespace ozaki
