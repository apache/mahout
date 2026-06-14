---
title: Experimental Neutrosophic Operator Layer for QuMat
sidebar_label: Experimental Neutrosophic Operator Layer
---

# RFC: Experimental Neutrosophic Operator Layer for QuMat

Status: phase-1 foundation draft  
Related issue: [apache/mahout#1372](https://github.com/apache/mahout/issues/1372)

## Context and Motivation

QuMat already provides a meaningful operator surface for standard quantum-circuit work:

- unified circuit construction
- standard gate application
- backend dispatch across `qiskit`, `cirq`, and `amazon_braket`
- public-facing positioning as Mahout's quantum-computing layer

At the same time, some research directions require a state carrier that is not reducible to a standard qubit. A conservative first step is to treat that need as a reference-model problem, not a backend-rewrite problem.

The immediate goal is therefore modest:

- define a narrow experimental direction
- keep it separate from production backend semantics
- anchor it in published sources before any broader implementation claims are made

## Problem Statement

Current QuMat supports standard qubit-oriented gates and backend execution, but it does not provide:

- a non-binary state carrier for indeterminacy-bearing operators
- an experimental namespace for non-standard operator families
- a structured comparison surface between standard qubit execution and a richer reference-state model

Without that, the following cannot be explored in a disciplined way:

- neutrobit semantics
- neutrosophic `W/X/Y/Z/H` gates over a `3x3` carrier
- comparison studies that preserve the distinction between:
  - qubit-compatible subspace behavior
  - genuinely `3`-state neutrosophic behavior

## Proposed Direction

The first increment should remain conservative and local-first.

Suggested shape:

- a `NeutroBit` reference carrier over `|0>`, `|1>`, and `|I>`
- a `NeutroGate` abstraction for named `3x3` operators
- a first operator family:
  - `W`
  - `X`
  - `Y`
  - `Z`
  - `H`
- local normalization and measurement utilities
- a projection and comparison harness against standard QuMat qubit circuits where comparison is mathematically valid

The first goal is observability and reference semantics, not backend replacement.

## Why QuMat Is a Good Candidate

QuMat is a reasonable host for this discussion because it already has:

- a unified operator-facing API
- documented standard gate semantics
- explicit backend modularity
- a live kernel-method direction under `MAHOUT-2200`

The fit is real at the level of:

- experimental operator abstraction
- comparison harnesses
- future data-encoding and kernel-method bridges
- non-breaking extension points

The fit is not yet direct at the level of:

- native `3`-state execution on existing QuMat backends
- immediate upstream implementation
- claims that Mahout already supports neutrosophic carriers

## Academic and Technical Foundation

The direct source pack for this direction is documented in:

- [Neutrosophic operator-layer bibliography](../academic/neutrosophic-operator-layer-bibliography.md)
- [Implementation-to-citation map](../academic/implementation-to-citation-map.md)

Those documents define what is:

- source-backed
- local design extension
- unsupported

## Proposed Phased Implementation

### Phase 1 - Reference kernel

Define a local reference-state kernel with:

- carrier semantics
- operator semantics
- normalization rules
- deterministic tests

### Phase 2 - Gate family

Add the first experimental operator family:

- `W`
- `X`
- `Y`
- `Z`
- `H`

### Phase 3 - Comparison harness

Add a constrained comparison surface between:

- standard QuMat qubit execution
- projected behavior from the experimental reference carrier

### Phase 4 - Kernel and data-encoding fit analysis

Only after the reference layer is stable, evaluate whether the direction can connect conservatively to Mahout's kernel-method and data-encoding work.

## Explicit Constraints

This RFC does not propose:

- rewriting QuMat backend modules in the first pass
- changing current production backend semantics
- claiming native neutrobit execution on `qiskit`, `cirq`, or `amazon_braket`
- replacing Mahout's current architecture with a new theory-first model

## Open Questions

1. Should a future public proposal stay strictly at the operator-abstraction level?
2. Should the first comparison harness stay basis-state only?
3. If kernel-method work becomes the strongest fit, should the operator layer remain purely experimental while only the encoding bridge is discussed publicly?

## Maintainer-Safe Ask

Would QuMat accept a narrowly scoped experimental operator-extension discussion that is:

- explicitly separate from production backend semantics
- local-reference-first
- comparison-oriented before implementation-oriented
- grounded in published academic sources
