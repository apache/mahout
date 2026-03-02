---
title: QuMat Documentation
---

![QuMat Logo](assets/mascot_with_text.png)

# QuMat

QuMat is a high level Python library for interfacing with multiple
quantum computing backends. It is designed to be easy to use and to abstract
the particularities of each backend, so that you may 'write once, run
anywhere.'

## Documentation

### Getting Started
- [Getting Started with QuMat](./qumat/getting-started) - Introduction and setup guide

### Core Concepts
- [Basic Gates](./qumat/basic-gates) - Introduction to fundamental quantum gates (NOT, Hadamard, CNOT, Toffoli, SWAP, Pauli gates, CSWAP, U gate)
- [Parameterized Quantum Circuits and Rotation Gates](./qumat/parameterized-circuits) - Rotation gates (Rx, Ry, Rz) and creating/optimizing parameterized circuits

### API Reference
- [API Documentation](./qumat/api) - Complete reference for all QuMat class methods

### Additional Resources
- [Parameterized Quantum Circuits: Developer's Guide](./advanced/pqc) - In-depth guide to PQCs
- [Qumat Gap Analysis for PQC](./advanced/gap-analysis) - Analysis of PQC capabilities

### Qumat Components
- [Qumat (Circuits)](./qumat) - Quantum circuit abstraction layer
- [QDP (Quantum Data Plane)](./qdp) - GPU-accelerated data encoding
