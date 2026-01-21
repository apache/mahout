---
title: Quickstart
sidebar_label: Quickstart
---

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

# Qumat Quickstart

## Installation

Install Qumat using pip:

```bash
pip install qumat
```

## Basic Usage

```python
from qumat import QumatCircuit

# Create a simple quantum circuit
circuit = QumatCircuit(2)
circuit.h(0)      # Hadamard gate on qubit 0
circuit.cx(0, 1)  # CNOT gate

# Run the circuit
result = circuit.run()
print(result)
```

## Next Steps

- [Getting Started with Qumat](/docs/getting_started_with_qumat)
- [Basic Gates](/docs/basic_gates)
- [API Reference](/docs/api)

## Legacy Mahout MapReduce

:::note
The legacy Mahout MapReduce functionality is deprecated. For current quantum computing features, see the [Qumat documentation](/docs/qumat).
:::
