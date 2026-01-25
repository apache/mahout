---
title: QuMat Documentation
---

![QuMat Logo](assets/mascot.png)

# QuMat

QuMat is a high level Python library for interfacing with multiple
quantum computing backends. It is designed to be easy to use and to abstract
the particularities of each backend, so that you may 'write once, run
anywhere.'

## Documentation

### Getting Started
- [Getting Started with QuMat](./getting-started-with-qumat) - Introduction and setup guide

### Core Concepts
- [Basic Gates](./basic-gates) - Introduction to fundamental quantum gates (NOT, Hadamard, CNOT, Toffoli, SWAP, Pauli gates, CSWAP, U gate)
- [Parameterized Quantum Circuits and Rotation Gates](./parameterized-circuits) - Rotation gates (Rx, Ry, Rz) and creating/optimizing parameterized circuits

### API Reference
- [API Documentation](./api) - Complete reference for all QuMat class methods

### Additional Resources
- [Parameterized Quantum Circuits: Developer's Guide](./pqc) - In-depth guide to PQCs
- [Qumat Gap Analysis for PQC](./qumat-gap-analysis-for-pqc) - Analysis of PQC capabilities

### Qumat Subprojects
- [Qumat Core](./qumat) - Quantum circuit abstraction layer
- [QDP (Quantum Data Plane)](./qumat/qdp) - GPU-accelerated data encoding
