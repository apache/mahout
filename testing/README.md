
<!--
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Apache Mahout Testing Suite

This directory contains the unified test suite for Apache Mahout, covering both the Qumat quantum computing library and the QDP (Quantum Data Plane) GPU-accelerated components.

## Test Structure

```
testing/
├── qumat/              # Qumat backend tests
├── qdp/                # QDP tests (GPU-required)
├── utils/              # Shared test utilities
└── conftest.py         # Pytest configuration and fixtures
```

## To run QDP test

Before running QDP tests, build the `_qdp` module into the main venv:

```bash
uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
```

Then run the tests:

```bash
uv run pytest testing/qdp/ -v
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only Qumat tests (works without GPU)
pytest testing/qumat/

# Run only QDP tests (requires GPU and built extension)
pytest testing/qdp/

# Run specific test file
pytest testing/qumat/test_create_circuit.py

# Run tests matching a pattern
pytest -k "hadamard"

# Run tests for a specific backend
pytest -k "qiskit" testing/qumat/
```

## QDP Test Handling

QDP-related tests are automatically skipped when the QDP extension or a compatible GPU is not available.

- QDP tests do **not** block running the Qumat test suite
- Contributors without CUDA GPUs can run all Qumat tests normally
- QDP tests are only executed when the required extension and hardware are available
