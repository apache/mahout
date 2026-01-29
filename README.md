
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

# Apache Mahout

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://www.python.org/)
[![GitHub Stars](https://img.shields.io/github/stars/apache/mahout.svg)](https://github.com/apache/mahout/stargazers)
[![GitHub Contributors](https://img.shields.io/github/contributors/apache/mahout.svg)](https://github.com/apache/mahout/graphs/contributors)

The goal of the Apache Mahout™ project is to build an environment for quickly creating scalable, performant machine learning applications.\
For additional information about Mahout, visit the [Mahout Home Page](http://mahout.apache.org/)

## Qumat

<p align="center">
  <img src="https://raw.githubusercontent.com/apache/mahout/refs/heads/main/docs/assets/mascot_with_text.png" width="400" alt="Apache Mahout">
</p>

Qumat is a high-level Python library for quantum computing that provides:

- **Quantum Circuit Abstraction** - Build quantum circuits with standard gates (Hadamard, CNOT, Pauli, etc.) and run them on Qiskit, Cirq, or Amazon Braket with a single unified API. Write once, execute anywhere. Check out [basic gates](docs/basic-gates.md) for a quick introduction to the basic gates supported across all backends.
- **QDP (Quantum Data Plane)** - Encode classical data into quantum states using GPU-accelerated kernels. Zero-copy tensor transfer via DLPack lets you move data between PyTorch, NumPy, and TensorFlow without overhead.

## Quick Start

```bash
git clone https://github.com/apache/mahout.git
cd mahout
pip install uv
uv sync                     # Core Qumat
uv sync --extra qdp         # With QDP (requires CUDA GPU)
```

### Qumat: Run a Quantum Circuit

```python
from qumat import QuMat

qumat = QuMat({"backend_name": "qiskit", "backend_options": {"simulator_type": "aer_simulator"}})
qumat.create_empty_circuit(num_qubits=2)
qumat.apply_hadamard_gate(0)
qumat.apply_cnot_gate(0, 1)
qumat.execute_circuit()
```

### QDP: Encode data for Quantum ML

```python
import qumat.qdp as qdp

engine = qdp.QdpEngine(device_id=0)
qtensor = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2, encoding_method="amplitude")
```

## Roadmap

### 2024
- [x] Transition of Classic to maintenance mode
- [x] Integration of Qumat with hardened (tests, docs, CI/CD) Cirq, Qiskit, and Braket backends
- [x] Integration with Amazon Braket
- [x] [Public talk about Qumat](https://2024.fossy.us/schedule/presentation/265/)

### 2025
- [x] [FOSDEM talk](https://fosdem.org/2025/schedule/event/fosdem-2025-5298-introducing-qumat-an-apache-mahout-joint-/)
- [x] QDP: Foundation & Infrastructure (Rust workspace, build configuration)
- [x] QDP: Core Implementation (CUDA kernels, CPU preprocessing, GPU memory management)
- [x] QDP: Zero-copy and Safety (DLManagedTensor, DLPack structures)
- [x] QDP: Python Binding (PyO3 wrapping, DLPack protocol)

### Q1 2026
- [ ] QDP: Input Format Support (PyTorch, NumPy, TensorFlow integration)
- [ ] QDP: Verification and Testing (device testing, benchmarking)
- [ ] QDP: Additional Encoders (angle/basis encoding, multi-GPU optimization)
- [ ] QDP: Integration & Release (documentation, example notebooks, PyPI publishing)

## Legal
Please see the `NOTICE.txt` included in this directory for more information.
