
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

Welcome to Apache Mahout!
===========
The goal of the Apache Mahout™ project is to build an environment for quickly creating scalable, performant machine learning applications.

For additional information about Mahout, visit the [Mahout Home Page](http://mahout.apache.org/)

![Qumat Logo](docs/assets/mascot.png)

# Qumat

Qumat is a POC of a high level Python library for intefacing with multiple quantum computing backends. It is designed to be easy to use and to abstract the particularities of each backend, so that you may 'write once, run anywhere.' Like the Java of quantum computing, but Java is the new COBOL so we're trying to distance ourselves from that comparison :P

Check out [basic gates](docs/basic_gates.md) for a quick introduction to the basic gates. These are now supported across multiple quantum computing frameworks, including Qiskit, Cirq, and Braket.

## Getting started

To install dependencies, run the following:
```
pip install uv
uv sync --group dev
```

### Quantum Data Plane (QDP)

QDP provides GPU-accelerated quantum state encoding with zero-copy PyTorch integration. To install with QDP support:

```bash
uv sync --extra qdp
```

Usage:
```python
import qumat.qdp as qdp

engine = qdp.QdpEngine(device_id=0)
qtensor = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2, encoding_method="amplitude")

# Zero-copy transfer to PyTorch
import torch
torch_tensor = torch.from_dlpack(qtensor)
```

Note: QDP requires a CUDA-capable GPU.

## Roadmap

### Q2 2024
- [x] Transition of Classic to maintenance mode

### Q3 2024
- [ ] Integration of Qumat with hardened (tests, docs, CI/CD) Cirq, Qiskit, and Braket backends
- [ ] Initiation of kernel methods
- [x] Integration with Amazon Braket
- [x] [Public talk about Qumat](https://2024.fossy.us/schedule/presentation/265/)

### Q3 2024
- [ ] Development of distributed quantum solvers

### Q1 2025
- [x] [FOSDEM talk](https://fosdem.org/2025/schedule/event/fosdem-2025-5298-introducing-qumat-an-apache-mahout-joint-/)

#### Legal
Please see the `NOTICE.txt` included in this directory for more information.
