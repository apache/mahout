---
title: QDP Input Format Architecture
sidebar_label: Readers
---

# QDP Input Format Architecture

This document describes the refactored input handling system in QDP that makes it easy to support multiple data formats.

## Overview

QDP now uses a trait-based architecture for reading quantum data from various sources. This design allows adding new input formats (NumPy, PyTorch, HDF5, etc.) without modifying core library code.

## Architecture

### Core Traits

#### `DataReader` Trait
Basic interface for batch reading:
```rust
pub trait DataReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)>;
    fn get_sample_size(&self) -> Option<usize> { None }
    fn get_num_samples(&self) -> Option<usize> { None }
}
```

#### `StreamingDataReader` Trait
Extended interface for large files that don't fit in memory:
```rust
pub trait StreamingDataReader: DataReader {
    fn read_chunk(&mut self, buffer: &mut [f64]) -> Result<usize>;
    fn total_rows(&self) -> usize;
}
```

### Implemented Formats

| Format | Reader | Streaming | Status |
|--------|--------|-----------|--------|
| Parquet | `ParquetReader` | ✅ `ParquetStreamingReader` | ✅ Complete |
| Arrow IPC | `ArrowIPCReader` | ❌ | ✅ Complete |
| NumPy | `NumpyReader` | ❌ | ✅ Complete |
| PyTorch | `TorchReader` | ❌ | ✅ (feature: `pytorch`) |

### NumPy memory guard (`QDP_NPY_MAX_BYTES`)

`NumpyReader` loads the whole `.npy` array into host memory. To fail fast
instead of being killed by the OOM killer, it first parses only the file
header (shape, dtype, `fortran_order`) and estimates the array size as
`prod(shape) * itemsize`. The load is refused with a
`MahoutError::MemoryAllocation` error (a `RuntimeError` in Python) when the
estimate is over the limit.

The limit is controlled by the `QDP_NPY_MAX_BYTES` environment variable:

| Value | Behaviour |
|-------|-----------|
| unset, empty, `auto` | Refuse when `estimate * 2 > 0.8 * MemAvailable` |
| `17179869184`, `16G`, `512MiB` | Refuse when `estimate > value` |
| `off`, `none`, `unlimited`, `disabled` | No check |

Notes:

- Suffixes are binary and case-insensitive: `K`, `M`, `G`, `T`, optionally
  followed by `B` or `iB` (`16G` = `16GiB` = 17179869184 bytes). `0` and
  unparseable values are errors, so a typo cannot silently turn the guard off.
- The `2x` in `auto` mode allows for a second copy of the data (for example
  flattening a Fortran-order array to row-major).
- `MemAvailable` comes from `/proc/meminfo` on Linux. On other platforms, or
  if it cannot be read, `auto` logs a warning and allows the load. Set an
  explicit byte limit if you need the guard there.
- Inside a container, `MemAvailable` is the host's figure, not the cgroup
  limit. Set an explicit limit if your container has a smaller memory cap.
- Independently of the limit, the reader checks that the file is at least as
  long as the header claims, so a truncated or corrupt file is rejected before
  any large allocation, even with the guard set to `off`.
- Rust callers can bypass the environment variable with
  `NumpyReader::new(path)?.with_memory_limit(NpyMemoryLimit::MaxBytes(n))`.

```bash
# Allow up to 64 GiB regardless of currently available memory
export QDP_NPY_MAX_BYTES=64G
# Turn the check off entirely
export QDP_NPY_MAX_BYTES=off
```

For datasets that do not fit in memory, convert them to Parquet and use the
streaming reader (`ParquetStreamingReader` in Rust, or
`QuantumDataLoader(...).source_file(path, streaming=True)` in Python).

## Benefits

### 1. Easy Extension
Adding a new format requires only:
- Implementing the `DataReader` trait
- Registering in `readers/mod.rs`
- Optional: Add convenience functions

No changes to core QDP code needed!

### 2. Zero Performance Overhead
- Traits use static dispatch where possible
- No runtime polymorphism overhead in hot paths
- Same zero-copy and streaming capabilities as before
- No memory allocation overhead

### 3. Backward Compatibility
All existing APIs continue to work:
```rust
// Old API still works
let (data, samples, size) = read_parquet_batch("data.parquet")?;
let (data, samples, size) = read_arrow_ipc_batch("data.arrow")?;

// ParquetBlockReader is now an alias to ParquetStreamingReader
let mut reader = ParquetBlockReader::new("data.parquet", None)?;
reader.read_chunk(&mut buffer)?;
```

### 4. Polymorphic Usage
Readers can be used generically:
```rust
fn process_data<R: DataReader>(mut reader: R) -> Result<()> {
    let (data, samples, size) = reader.read_batch()?;
    // Process data...
}

// Works with any reader!
process_data(ParquetReader::new("data.parquet", None)?)?;
process_data(ArrowIPCReader::new("data.arrow")?)?;
```

## Usage Examples

### Basic Reading

```rust
use qdp_core::reader::DataReader;
use qdp_core::readers::ArrowIPCReader;

let mut reader = ArrowIPCReader::new("quantum_states.arrow")?;
let (data, num_samples, sample_size) = reader.read_batch()?;

println!("Read {} samples of {} qubits",
         num_samples, (sample_size as f64).log2() as usize);
```

### Streaming Large Files

```rust
use qdp_core::reader::StreamingDataReader;
use qdp_core::readers::ParquetStreamingReader;

let mut reader = ParquetStreamingReader::new("large_dataset.parquet", None)?;
let mut buffer = vec![0.0; 1024 * 1024]; // 1M element buffer

loop {
    let written = reader.read_chunk(&mut buffer)?;
    if written == 0 { break; }

    // Process chunk
    process_chunk(&buffer[..written])?;
}
```

### Format Detection

```rust
fn read_quantum_data(path: &str) -> Result<(Vec<f64>, usize, usize)> {
    use qdp_core::reader::DataReader;

    if path.ends_with(".parquet") {
        ParquetReader::new(path, None)?.read_batch()
    } else if path.ends_with(".arrow") {
        ArrowIPCReader::new(path)?.read_batch()
    } else if path.ends_with(".npy") {
        NumpyReader::new(path)?.read_batch()
    } else if path.ends_with(".pt") || path.ends_with(".pth") {
        TorchReader::new(path)?.read_batch()
    } else {
        Err(MahoutError::InvalidInput("Unsupported format".into()))
    }
}
```

## Adding New Formats

See ADDING_INPUT_FORMATS.md (TODO) for detailed instructions.

Quick overview:
1. Create `readers/myformat.rs`
2. Implement `DataReader` trait
3. Add to `readers/mod.rs`
4. Add tests
5. (Optional) Add convenience functions

## File Organization

```
qdp-core/src/
├── reader.rs              # Trait definitions
├── readers/
│   ├── mod.rs            # Reader registry
│   ├── parquet.rs        # Parquet implementation
│   ├── arrow_ipc.rs      # Arrow IPC implementation
│   ├── numpy.rs          # NumPy implementation
│   └── torch.rs          # PyTorch (feature-gated)
├── io.rs                 # Legacy API & helper functions
└── lib.rs                # Main library

examples/
└── flexible_readers.rs   # Demo of architecture

docs/
├── readers/
│   └── README.md         # This file
└── ADDING_INPUT_FORMATS.md  # Extension guide
```

## Performance Considerations

### Memory Efficiency
- **Parquet Streaming**: Constant memory usage for any file size
- **Zero-copy**: Direct buffer access where possible
- **Pre-allocation**: Reserves capacity when total size is known

### Speed
- **Static dispatch**: No virtual function overhead
- **Batch operations**: Minimizes function call overhead
- **Efficient formats**: Columnar storage (Parquet/Arrow) for fast reading

### Benchmarks
The architecture maintains the same performance as before:
- Parquet streaming: ~2GB/s throughput
- Arrow IPC: ~4GB/s throughput (zero-copy)
- Memory usage: O(buffer_size), not O(file_size)

## Migration Guide

### For Users
No changes required! All existing code continues to work.

### For Contributors
If you were directly using internal reader structures:

**Before:**
```rust
let reader = ParquetBlockReader::new(path, None)?;
```

**After:**
```rust
// Still works (it's a type alias)
let reader = ParquetBlockReader::new(path, None)?;

// Or use the new name
let reader = ParquetStreamingReader::new(path, None)?;
```

## Future Enhancements

Planned format support:
- **NumPy streaming**: Chunked reads for large `.npy` files
- **PyTorch streaming**: Streaming support for large tensors
- **HDF5** (`.h5`): Scientific data storage
- **JSON**: Human-readable format for small datasets
- **CSV**: Simple tabular data

## Questions?

- See examples: `cargo run --example flexible_readers`
- Read extension guide: ADDING_INPUT_FORMATS.md (TODO)
- Check tests: `qdp-core/tests/*_io.rs`
