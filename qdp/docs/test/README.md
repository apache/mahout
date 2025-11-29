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
