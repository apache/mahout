# NVTX Profiling Guide

## Overview

NVTX (NVIDIA Tools Extension) provides performance markers visible in Nsight Systems. This project uses zero-cost macros that compile to no-ops when the `observability` feature is disabled.

## Run the NVTX Example (New Workflow)

Default builds exclude NVTX for zero overhead. The example below uses the
async pipeline workload (large input) to surface the new pipeline markers.

```bash
cd mahout/qdp
cargo run -p qdp-core --example nvtx_profile --features observability --release
```

**Expected output:**
```
=== NVTX Profiling Example ===

✓ Engine initialized
✓ Created test data: 262144 elements

Starting encoding (NVTX markers will appear in Nsight Systems)...
Expected NVTX markers:
  - Mahout::Encode
  - CPU::L2Norm
  - GPU::Alloc
  - GPU::H2DCopy
  - GPU::CopyEventRecord
  - GPU::H2D_Stage
  - GPU::Kernel
  - GPU::ComputeSync

✓ Encoding succeeded
✓ DLPack pointer: 0x558114be6250
✓ Memory freed

=== Test Complete ===
```

## Profile with Nsight Systems

Focus capture on the `Mahout::Encode` range (recommended):

```bash
nsys profile --trace=cuda,nvtx --capture-range=nvtx \
  --nvtx-capture=Mahout::Encode --force-overwrite=true -o nvtx-workflow \
  cargo run -p qdp-core --example nvtx_profile --features observability --release
```

This generates `nvtx-workflow.nsys-rep` and `nvtx-workflow.sqlite`.

## Viewing Results

### GUI View (Nsight Systems)

Open the report in Nsight Systems GUI:

```bash
nsys-ui report.nsys-rep
```

In the GUI timeline view, you will see:
- Colored blocks for each NVTX marker
- CPU timeline showing `CPU::L2Norm`
- GPU timeline showing `GPU::Alloc`, `GPU::H2DCopy`, `GPU::CopyEventRecord`, `GPU::H2D_Stage`, `GPU::Kernel`, `GPU::ComputeSync`
- Overall workflow covered by `Mahout::Encode`

### Command Line Statistics

NVTX range summary:

```bash
nsys stats --report nvtx_sum nvtx-workflow.nsys-rep
```

Note: very short pipeline ranges may be easier to verify in the GUI timeline.

## NVTX Markers

The following markers are tracked:

- `Mahout::Encode` - Complete encoding workflow
- `CPU::L2Norm` - L2 normalization on CPU
- `GPU::Alloc` - GPU memory allocation
- `GPU::H2DCopy` - Host-to-device memory copy
- `GPU::Kernel` - Kernel execution

The following pipeline ranges are also used where applicable:

- `GPU::CopyEventRecord` - Record copy completion event
- `GPU::H2D_Stage` - Host staging copy into pinned buffer
- `GPU::ComputeSync` - Compute stream synchronization

## Using Profiling Macros

The project provides zero-cost macros in `qdp-core/src/profiling.rs`:

```rust
// Profile a scope (automatically pops on exit)
crate::profile_scope!("MyOperation");

// Mark a point in time
crate::profile_mark!("Checkpoint");
```

When `observability` feature is disabled, these macros compile to no-ops with zero runtime cost.

## Example Location

Source code: `qdp-core/examples/nvtx_profile.rs`

## Troubleshooting

**NVTX markers not appearing:**
- Ensure `--features observability` is used during build
- Verify CUDA device is available
- Check that encoding actually executes

**nsys warnings:**
Warnings about CPU sampling are normal and can be ignored. They do not affect NVTX marker recording.

## Official Docs

- CUDA Runtime Profiler Control:
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html
- Nsight Systems User Guide (v2026.1):
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Compute Profiling Guide (latest as of now):
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
