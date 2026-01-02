![QuMat Logo](assets/mascot.png)

# QuMat

QuMat is a POC of a high level Python library for intefacing with multiple
quantum computing backends. It is designed to be easy to use and to abstract
the particularities of each backend, so that you may 'write once, run
anywhere.' Like the Java of quantum computing, but Java is the new COBOL so
we're trying to distance ourselves from that comparison :P

## Documentation

### Getting Started
- [Getting Started with QuMat](getting_started_with_qumat.md) - Introduction and setup guide

### Core Concepts
- [Basic Gates](basic_gates.md) - Introduction to fundamental quantum gates (NOT, Hadamard, CNOT, Toffoli, SWAP, Pauli gates, CSWAP, U gate)
- [Parameterized Quantum Circuits and Rotation Gates](parameterized_circuits.md) - Rotation gates (Rx, Ry, Rz) and creating/optimizing parameterized circuits

### API Reference
- [API Documentation](api.md) - Complete reference for all QuMat class methods

### Additional Resources
- [Parameterized Quantum Circuits: Developer's Guide](p_q_c.md) - In-depth guide to PQCs
- [Qumat Gap Analysis for PQC](qumat_gap_analysis_for_pqc.md) - Analysis of PQC capabilities

## Examples

You can also check out examples in the [examples](examples) directory. The
cool one is [quantum teleportation](examples/quantum_teleportation.py).
