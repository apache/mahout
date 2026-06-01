---
title: Implementation to Citation Map
sidebar_label: Implementation to Citation Map
---

# Implementation to Citation Map

Status: phase-1 citation gate  
Related issue: [apache/mahout#1372](https://github.com/apache/mahout/issues/1372)

## Rule

Every planned implementation should be in one of three states:

1. `source-backed`
2. `local design extension`
3. `unsupported`

Anything still marked `unsupported` should not be presented as if it were ready for implementation.

## Mapping

### NeutroBit carrier over `|0>`, `|1>`, `|I>`

- Status: `source-backed`
- Primary source:
  - `Neutrosophic Logic Based Quantum Computing`

### Neutrosophic `W` gate

- Status: `source-backed`
- Primary source:
  - `Neutrosophic Logic Based Quantum Computing`

### Neutrosophic `X/Y/Z` family

- Status: `source-backed`
- Primary source:
  - `Neutrosophic Logic Based Quantum Computing`
- Note:
  - software class shape remains an implementation choice

### Neutrosophic Hadamard

- Status: `source-backed`
- Primary source:
  - `Neutrosophic Logic Based Quantum Computing`

### `NeutroGate` software abstraction

- Status: `local design extension`
- Grounding:
  - justified by QuMat's gate-oriented API
- Note:
  - this is a software abstraction, not a paper-defined class contract

### Projection and comparison harness

- Status: `local design extension`
- Grounding:
  - needed to compare a `3`-state reference model against current QuMat qubit semantics

### Direct execution on `qiskit` / `cirq` / `amazon_braket` as neutrobit backends

- Status: `unsupported`
- Reason:
  - current repo and current published sources do not justify claiming native `3`-state execution on existing QuMat backends

### Kernel-method bridge into Mahout

- Status: `local design extension`
- Grounding:
  - `MAHOUT-2200` confirms kernel-method direction exists
- Missing:
  - exact technical bridge remains unproven and should be treated as tentative, not oversold
