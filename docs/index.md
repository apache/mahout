---
title: Mahout Documentation
sidebar_label: Mahout Documentation
---

![Qumat Logo](assets/mascot_with_text.png)

# Apache Mahout Documentation

Apache Mahout is an Apache Software Foundation project for quantum computing in Python. It ships two components:

- **Qumat** is a high-level library for building quantum circuits with standard gates and running them on Qiskit, Cirq, or Amazon Braket through one unified API. Write once, execute anywhere.
- **QDP (Quantum Data Plane)** is a GPU-accelerated engine for encoding classical data into quantum states, with zero-copy tensor transfer to and from PyTorch, NumPy, and TensorFlow via DLPack.

New here? Start with [Getting Started](./qumat/getting-started), which covers installation and your first circuit.

## Qumat (Circuits)

- [Overview](./qumat) - What Qumat is and which backends it supports
- [Basic Gates](./qumat/basic-gates) - NOT, Hadamard, CNOT, Toffoli, SWAP, Pauli, CSWAP, and U gates
- [Parameterized Circuits](./qumat/parameterized-circuits) - Rotation gates (Rx, Ry, Rz) and building circuits with tunable parameters
- [API Reference](./qumat/api) - Generated reference for the `qumat` package
- [Core Concepts](./qumat/concepts) - The ideas behind the circuit abstraction
- [Examples](./qumat/examples) - Worked examples across backends

## QDP (Data Encoding)

- [Overview](./qdp) - What QDP is and how it fits with Qumat
- [Getting Started with QDP](./qdp/getting-started) - Installation, GPU requirements, and a first encoding
- [Core Concepts](./qdp/concepts) - Encoding methods, tensors, and the execution model
- [API Reference](./qdp/api) - Generated reference for the `qumat_qdp` package
- [Python API](./qdp/python-api) - The user-facing Python facade, loaders, and backend selection
- [Examples](./qdp/examples) - End-to-end encoding examples
- Internals: [Readers](./qdp/readers), [Observability](./qdp/observability), [Testing](./qdp/testing)

## Resources

- [PQC Guides](./advanced) - Developer's guide and gap analysis for parameterized quantum circuits
- [Quantum Computing Primer](./learning/quantum-computing-primer) - A ten-chapter introduction to quantum computing using Qumat
- [Research Papers](./learning/papers) - Papers and publications related to Qumat and quantum computing
- [Books, Tutorials and Talks](./learning/books-tutorials-and-talks) - Reading, articles, lectures, and conference talks
- [Reference Reading](./learning/reference-reading) - Background material on linear algebra, statistics, and machine learning
- [Professional Support](./learning/professional-support) - People and companies offering Mahout support and talks

## Community

- [Community Overview](./community) - How the project communicates and makes decisions
- [Who We Are](./community/who-we-are) - PMC members and committers
- [Mailing Lists](./community/mailing-lists) - Subscribe to the user, dev, and commits lists
- [How to Contribute](./about/how-to-contribute) - Getting involved and submitting changes
- [PR Policy and Review Guidelines](./community/pr-policy-and-review-guidelines) - What reviewers expect
- [Code of Conduct](./community/code-of-conduct) - Community standards
