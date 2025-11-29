# NVTX Profiling Guide

## Overview

NVTX (NVIDIA Tools Extension) provides performance markers visible in Nsight Systems. This project uses zero-cost macros that compile to no-ops when the `observability` feature is disabled.

## Build with NVTX

Default builds exclude NVTX for zero overhead. Enable profiling with:

```bash
cd mahout/qdp
cargo build -p qdp-core --example nvtx_profile --features observability --release
```

## Run Example

```bash
./target/release/examples/nvtx_profile
```

**Expected output:**
```
=== NVTX Profiling Example ===

✓ Engine initialized
✓ Created test data: 1024 elements

Starting encoding (NVTX markers will appear in Nsight Systems)...
Expected NVTX markers:
  - Mahout::Encode
  - CPU::L2Norm
  - GPU::Alloc
  - GPU::H2DCopy
  - GPU::KernelLaunch
  - GPU::Synchronize
  - DLPack::Wrap

✓ Encoding succeeded
✓ DLPack pointer: 0x558114be6250
✓ Memory freed

=== Test Complete ===
```

## Profile with Nsight Systems

```bash
nsys profile --trace=cuda,nvtx -o report ./target/release/examples/nvtx_profile
```

This generates `report.nsys-rep` and `report.sqlite`.

## Viewing Results

### GUI View (Nsight Systems)

Open the report in Nsight Systems GUI:

```bash
nsys-ui report.nsys-rep
```

In the GUI timeline view, you will see:
- Colored blocks for each NVTX marker
- CPU timeline showing `CPU::L2Norm`
- GPU timeline showing `GPU::Alloc`, `GPU::H2DCopy`, `GPU::Kernel`
- Overall workflow covered by `Mahout::Encode`

### Command Line Statistics

View summary statistics:

```bash
nsys stats report.nsys-rep
```

**Example NVTX Range Summary output:**
```
Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)   Style        Range     
--------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  --------  --------------
   50.0       11,207,505          1  11,207,505.0  11,207,505.0  11,207,505  11,207,505          0.0  StartEnd  Mahout::Encode
   48.0       10,759,758          1  10,759,758.0  10,759,758.0  10,759,758  10,759,758          0.0  StartEnd  GPU::Alloc    
    1.8          413,753          1     413,753.0     413,753.0     413,753     413,753          0.0  StartEnd  CPU::L2Norm   
    0.1           15,873          1      15,873.0      15,873.0      15,873      15,873          0.0  StartEnd  GPU::H2DCopy  
    0.0              317          1         317.0         317.0         317         317          0.0  StartEnd  GPU::KernelLaunch
```

The output shows:
- Time percentage for each operation
- Total time in nanoseconds
- Number of instances
- Average, median, min, max execution times

**CUDA API Summary** shows detailed CUDA call statistics:

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)          Name        
 --------  ---------------  ---------  -----------  -----------  --------  ----------  -----------  --------------------
     99.2       11,760,277          2  5,880,138.5  5,880,138.5     2,913  11,757,364  8,311,652.0  cuMemAllocAsync     
      0.4           45,979          2     22,989.5     22,989.5     7,938      38,041     21,286.0  cuMemcpyHtoDAsync_v2
      0.1           14,722          1     14,722.0     14,722.0    14,722      14,722          0.0  cuEventCreate       
      0.1           13,100          3      4,366.7      3,512.0       861       8,727      4,002.0  cuStreamSynchronize 
      0.1            9,468         11        860.7        250.0       114       4,671      1,453.3  cuCtxSetCurrent     
      0.1            6,479          1      6,479.0      6,479.0     6,479       6,479          0.0  cuEventDestroy_v2   
      0.0            4,599          2      2,299.5      2,299.5     1,773       2,826        744.6  cuMemFreeAsync  
- Memory allocation (`cuMemAllocAsync`)
- Memory copies (`cuMemcpyHtoDAsync_v2`)
- Stream synchronization (`cuStreamSynchronize`)

## NVTX Markers

The following markers are tracked:

- `Mahout::Encode` - Complete encoding workflow
- `CPU::L2Norm` - L2 normalization on CPU
- `GPU::Alloc` - GPU memory allocation
- `GPU::H2DCopy` - Host-to-device memory copy
- `GPU::KernelLaunch` - CPU-side kernel launch
- `GPU::Synchronize` - CPU waiting for GPU completion
- `DLPack::Wrap` - Conversion to DLPack pointer

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
