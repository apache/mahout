# QDP Core Test Suite

Unit tests for QDP core library covering input validation, API workflows, and memory safety.

## Test Files

### `validation.rs` - Input Validation

- Invalid encoder strategy rejection
- Qubit size validation (mismatch, zero, max limit 30)
- Empty and zero-norm data rejection
- Error type formatting
- Non-Linux platform graceful failure

### `api_workflow.rs` - API Workflow

- Engine initialization
- Amplitude encoding with DLPack pointer management

### `memory_safety.rs` - Memory Safety

- Memory leak detection (100 encode/free cycles)
- Concurrent state vector management
- DLPack tensor metadata validation

### `examples/dataloader_throughput.rs` - DataLoader Batch Throughput

- Simulates a QML training loop that streams batches of 64 vectors
- Producer/consumer model with configurable prefetch to avoid GPU starvation
- Reports vectors-per-second to verify QDP keeps the GPU busy
- Run: `cargo run -p qdp-core --example dataloader_throughput --release`
- Environment overrides: `BATCHES=<usize>` (default 200), `PREFETCH=<usize>` (default 16)

### `common/mod.rs` - Test Utilities

- `create_test_data(size)`: Generates normalized test data

## Running Tests

```bash
# Run all tests
cargo test --package qdp-core

# Run specific test file
cargo test --package qdp-core --test validation
cargo test --package qdp-core --test api_workflow
cargo test --package qdp-core --test memory_safety
```

## Requirements

- Linux OS (tests skip on other platforms)
- CUDA-capable GPU (tests skip if unavailable)
- Rust toolchain with CUDA support
