---
title: API Documentation Guide
sidebar_label: Documenting APIs
sidebar_position: 2
---

# API Documentation Guide

This guide explains how to write and maintain API documentation for Apache Mahout. Good documentation helps users understand and use our libraries effectively.

## Overview

Apache Mahout uses automated documentation generation:

- **Python**: [pdoc](https://pdoc.dev) generates documentation from docstrings
- **Rust**: [rustdoc](https://doc.rust-lang.org/rustdoc/) generates documentation from doc comments

Documentation is automatically built and deployed to the website when changes are merged to the main branch.

## Python Docstring Style

We use **reStructuredText (RST)** format for Python docstrings, which is compatible with both Sphinx and pdoc. All public classes, methods, and functions must have docstrings.

### Module Docstrings

Every Python module should have a docstring explaining its purpose:

```python
"""Quantum circuit abstraction layer.

This module provides a unified interface for quantum circuit programming
across multiple backends including Qiskit, Cirq, and Amazon Braket.
"""
```

### Class Docstrings

```python
class QuMat:
    """Unified quantum circuit interface.

    Provides a backend-agnostic API for creating and executing quantum
    circuits. Supports parameterized circuits for variational algorithms.

    :param backend_config: Configuration dictionary specifying the backend.
    :type backend_config: dict

    Example::

        >>> qc = QuMat({"backend_name": "qiskit", "backend_options": {...}})
        >>> qc.create_empty_circuit(2)
        >>> qc.apply_hadamard_gate(0)
    """
```

### Method Docstrings

```python
def apply_cnot_gate(self, control_qubit_index, target_qubit_index):
    """Apply a Controlled-NOT gate between two qubits.

    The CNOT gate flips the target qubit if the control qubit is |1⟩.
    This is fundamental for creating entanglement between qubits.

    :param control_qubit_index: Index of the control qubit.
    :type control_qubit_index: int
    :param target_qubit_index: Index of the target qubit.
    :type target_qubit_index: int
    :raises RuntimeError: If the circuit is not initialized.
    :raises ValueError: If qubit indices are out of range or equal.

    Example::

        >>> qc.apply_cnot_gate(0, 1)  # Entangle qubits 0 and 1
    """
```

### Required Documentation Elements

All public API elements must document:

| Element | Description |
|---------|-------------|
| **Purpose** | What does it do? (first line) |
| **Parameters** | Name, type, and description using `:param:` and `:type:` |
| **Returns** | Type and description using `:returns:` and `:rtype:` |
| **Raises** | Exceptions that may be raised using `:raises:` |
| **Examples** | Usage examples in `Example::` blocks (optional but encouraged) |

## Rust Documentation Style

We use standard rustdoc conventions with `///` doc comments for public items.

### Module Documentation

```rust
//! GPU-accelerated quantum state encoding.
//!
//! This module provides the core encoding engine that transforms classical
//! data into quantum states on NVIDIA GPUs.
```

### Struct/Enum Documentation

```rust
/// GPU encoder for quantum state preparation.
///
/// Manages CUDA context and dispatches encoding operations to the GPU.
/// Supports multiple precision modes and encoding strategies.
///
/// # Examples
///
/// ```
/// let engine = QdpEngine::new(0)?;
/// let tensor = engine.encode(&data, 4, "amplitude")?;
/// ```
pub struct QdpEngine {
    // ...
}
```

### Function Documentation

```rust
/// Encode classical data into a quantum state.
///
/// Transforms input data into quantum state amplitudes using the specified
/// encoding method. Returns a DLPack-compatible tensor for zero-copy
/// integration with PyTorch.
///
/// # Arguments
///
/// * `data` - Input data slice (must be finite, non-empty)
/// * `num_qubits` - Number of qubits (1-30)
/// * `encoding_method` - Strategy: `"amplitude"`, `"angle"`, or `"basis"`
///
/// # Returns
///
/// DLPack pointer for GPU tensor access.
///
/// # Errors
///
/// Returns `MahoutError::InvalidInput` if:
/// - `num_qubits` is out of range
/// - Input contains NaN or Inf values
/// - Data length exceeds 2^num_qubits for amplitude encoding
///
/// # Safety
///
/// The returned pointer must not be freed manually. Ownership transfers
/// to the consumer via the DLPack protocol.
pub fn encode(&self, data: &[f64], num_qubits: usize, encoding_method: &str)
    -> Result<*mut DLManagedTensor>
```

### Required Rust Documentation

| Element | Syntax | When Required |
|---------|--------|---------------|
| Summary | First paragraph | Always |
| Description | Additional paragraphs | Complex items |
| Arguments | `# Arguments` section | Functions with params |
| Returns | `# Returns` section | Non-void functions |
| Errors | `# Errors` section | Fallible functions |
| Safety | `# Safety` section | Unsafe functions |
| Examples | `# Examples` section | Public APIs |
| Panics | `# Panics` section | If function can panic |

## Generating Documentation Locally

### Python

```bash
# Install pdoc if not already installed
pip install pdoc

# Generate documentation (from repository root)
pdoc --output-dir docs/api/python --docformat restructuredtext qumat

# Or use the provided script
./scripts/generate-python-docs.sh
```

### Rust

```bash
# Generate documentation (from qdp directory)
cd qdp
cargo doc --no-deps --package qdp-core --open

# Or use the provided script (from repository root)
./scripts/generate-rust-docs.sh
```

## CI Verification

API documentation is automatically verified on every PR that modifies Python or Rust source files. The CI will:

1. Generate documentation from source code
2. Verify output files are created successfully
3. Build the complete website with integrated API docs
4. Fail if documentation generation produces errors

### Triggering Doc Builds

Documentation CI runs when these paths change:

- `qumat/**` - Python Qumat library
- `qdp/qdp-core/**` - Rust QDP core
- `scripts/generate-*-docs.sh` - Doc generation scripts

## Adding Documentation for New APIs

When adding new public APIs:

1. **Write comprehensive docstrings/doc comments** following the style guides above
2. **Run doc generation locally** to verify formatting
3. **Check for warnings** - pdoc and rustdoc will warn about missing documentation
4. **Include examples** for non-trivial APIs

### Rust: Enabling Missing Docs Warnings

The `qdp-core` crate is configured to warn on missing documentation:

```toml
[lints.rust]
missing_docs = "warn"
```

To see warnings locally:

```bash
cd qdp
cargo doc --package qdp-core 2>&1 | grep "warning: missing documentation"
```

## Website Integration

Generated documentation is placed in:

| Type | Source Location | Website URL |
|------|-----------------|-------------|
| Python | `docs/api/python/` | `/docs/api/python` |
| Rust | `website/static/api/rust/` | `/api/rust/qdp_core/` |

The existing `sync-docs.js` script automatically picks up Python markdown files. Rust HTML documentation is served as static files.

## Tips for Good Documentation

1. **Start with the "what"** - First sentence should explain what the item does
2. **Explain the "why"** - Help users understand when to use it
3. **Show, don't just tell** - Include runnable examples
4. **Document edge cases** - What happens with empty input? Invalid parameters?
5. **Keep it current** - Update docs when behavior changes
6. **Use consistent terminology** - Match terms used elsewhere in the codebase
