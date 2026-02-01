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

// CUDA Kernel Configuration Header
//
// Centralized configuration constants for all CUDA kernels.
// This header ensures consistency across kernel implementations and
// simplifies maintenance and tuning of kernel launch parameters.

#ifndef KERNEL_CONFIG_H
#define KERNEL_CONFIG_H

// ============================================================================
// Block Size Configuration
// ============================================================================
// Default block size (threads per block)
// 256 threads = 8 warps, optimal for most modern GPUs (SM 7.0+)
// Provides good occupancy while maintaining low register pressure
#define DEFAULT_BLOCK_SIZE 256

// ============================================================================
// Grid Size Limits
// ============================================================================
// Maximum number of blocks for general kernel launches
// Limits grid size to avoid scheduler overhead on most GPUs
#define MAX_GRID_BLOCKS 2048

// Maximum number of blocks for L2 norm reduction kernels
// Used for kernels that process large input arrays
#define MAX_GRID_BLOCKS_L2_NORM 4096

// Maximum blocks per sample for batch processing
// Limits per-sample parallelism to maintain good load balancing
#define MAX_BLOCKS_PER_SAMPLE 32

// CUDA grid dimension limit for 1D launches (2^31 - 1, signed 32-bit int max)
// This is a hardware limitation, not a tunable parameter
#define CUDA_MAX_GRID_DIM_1D 2147483647

// ============================================================================
// Qubit Limits
// ============================================================================
// Maximum qubits supported (16GB GPU memory limit)
// This limit ensures state vectors fit within practical GPU memory constraints
#define MAX_QUBITS 30

// ============================================================================
// Convenience Macros
// ============================================================================
// Common block size alias for consistency
#define BLOCK_SIZE DEFAULT_BLOCK_SIZE

// Finalization kernel block size (typically same as default)
#define FINALIZE_BLOCK_SIZE DEFAULT_BLOCK_SIZE

#endif // KERNEL_CONFIG_H
