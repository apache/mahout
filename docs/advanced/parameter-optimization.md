---
title: "Design Note: Parameter Optimization Utilities"
sidebar_label: Parameter Optimization Design
---

# Design Note: Parameter Optimization Utilities

**Status:** proposal with a reference implementation (`qumat/optimize.py`).
**Issue:** [#749](https://github.com/apache/mahout/issues/749).

Reviewers asked for a design document before code, because this is a
large addition to Qumat. This page is that document. The accompanying
code is a reference implementation of the design so that it can be
reviewed concretely; any part of it can change.

## Motivation

The [PQC gap analysis](./gap-analysis) lists the following as missing:

- Part 1, section 4: no functionality for optimizing parameters.
- Part 2, section 1: a parameter-shift rule for exact gradients.
- Part 2, section 4: an optimization module with gradient-based and
  gradient-free algorithms.
- Part 2, section 5: expectation values of observables.

Today, users write their own loop: bind parameters, run the circuit,
turn counts into a cost, and call `scipy.optimize.minimize` or update
the values by hand.

## Goals

1. One call to optimize a parameterized circuit:
   `optimize_parameters(qumat, cost_fn, initial_params, method=...)`.
2. Exact gradients for every parameterized gate Qumat exposes
   (RX, RY, RZ), on every backend.
3. Exact, shot-free expectation values for tests and small simulations.
4. No new dependencies, and no changes to the `QuMat` class API.

## Non-goals

- Automatic differentiation through PyTorch, JAX or TensorFlow.
- Shot-noise-aware optimizers such as SPSA (a natural follow-up).
- General observables beyond products of Pauli Z.
- Hardware execution; the helpers use `get_final_state_vector()`.

## API

```python
from qumat import QuMat, expectation_z, optimize_parameters

qm = QuMat({"backend_name": "cirq", "backend_options": {}})
qm.create_empty_circuit(num_qubits=2)
qm.apply_ry_gate(0, "a")
qm.apply_cnot_gate(0, 1)
qm.apply_ry_gate(1, "c")

def cost(qumat, params):
    return expectation_z(qumat, params, 0) + expectation_z(qumat, params, 1)

result = optimize_parameters(qm, cost, {"a": 0.3, "c": 0.2},
                             method="adam", learning_rate=0.2)
result.params, result.cost, result.history, result.n_evaluations
```

| Function | Purpose |
|---|---|
| `expectation_z(qumat, params, qubits)` | Exact `<Z_i>` or `<Z_i Z_j ...>` |
| `parameter_shift_gradient(qumat, cost_fn, params)` | Exact gradient |
| `optimize_parameters(...)` | Run an optimizer, return `OptimizationResult` |

### The cost function contract

A cost function has the signature `cost_fn(qumat, params) -> float`.
It receives the `QuMat` instance and a `dict` of parameter values and
is responsible for binding them and evaluating the circuit. This shape
was chosen because:

- it lets the cost use any Qumat output (counts from
  `execute_circuit()`, or a state vector);
- the optimizer can call it with shifted values without knowing how
  the cost is computed;
- named parameters match how Qumat already identifies them.

Parameters not listed in `initial_params` must already be bound. They
stay fixed.

## Parameter-shift rule

Every rotation in Qumat has the form `U(t) = exp(-i t P / 2)` with `P`
a Pauli matrix, whose eigenvalues are ±1. If the cost is an expectation
value `f(t) = <psi| U(t)^† A U(t) |psi>` (or any other quantity linear
in the final state, such as a probability), then `f` is a sinusoid of
frequency 1 in `t`, `f(t) = a + b cos t + c sin t`, and

```text
f'(t) = [ f(t + pi/2) - f(t - pi/2) ] / 2
```

holds exactly, with no step size to choose. That is what
`parameter_shift_gradient` computes for each parameter.

### Shared parameters

Qumat identifies a parameter by name, so the same name may appear in
several gates. If one name drives `R` gates, shifting the name shifts
all `R` gates at once and `f` has frequencies up to `R`. The two-term
rule is then wrong. For `RY(a) RY(a) = RY(2a)` it gives

```text
[cos(2a + pi) - cos(2a - pi)] / 2 = 0   (true value: -2 sin 2a)
```

There are two standard fixes:

1. Shift each *occurrence* separately and add the results (product
   rule). This needs per-gate shifting, which Qumat's backends do not
   expose.
2. Use the general rule for a trigonometric polynomial of degree `R`
   (Wierichs, Izaac, Wang, Lin, *Quantum* 6, 677 (2022), eq. 16):

```text
f'(t) = sum_{mu=1..2R} f(t + x_mu) (-1)^(mu-1) / (4R sin^2(x_mu/2))
x_mu  = (2 mu - 1) pi / (2R)
```

The reference implementation uses option 2. Each backend gains a small
`count_parameter_occurrences(circuit)` function that reports `R` for
every parameter. The cost is `2R` evaluations per parameter, the same
as option 1.

### Non-linear costs

The rule needs a cost that is linear in the state. A squared error such
as `(<Z> - y)^2` is not. For those, either differentiate the linear part
and apply the chain rule yourself (the QML trainer in #1409 does this),
or pass `gradient="finite_difference"`.

## Qubit ordering

`get_final_state_vector()` returns the backend's native ordering:
Qiskit is little-endian (qubit 0 is the least significant bit of the
amplitude index), Cirq and Braket are big-endian. `expectation_z`
calls `canonical_state_vector()`, which converts Qiskit vectors to the
big-endian order so that `qubits=0` refers to the same qubit
everywhere. The tests check this with an `X` on qubit 0 of a 3-qubit
register on all three backends.

## Optimizers

| `method` | Kind | Gradient |
|---|---|---|
| `"gradient_descent"` | built in | `gradient` (default parameter shift) |
| `"adam"` | built in | `gradient` |
| any `scipy.optimize.minimize` method | SciPy | passed as `jac` where used |

The built-in methods are a few lines each and need no extra
dependencies. They stop after `max_iter` iterations or when the cost
changes by at most `tol`. SciPy is imported only when a SciPy method is
requested. It is already a hard dependency of `qiskit`, `qiskit-aer`
and `cirq-core`, so no dependency is added to `pyproject.toml`.

`OptimizationResult` reports the best parameters seen, their cost, the
cost history, the number of cost evaluations (including gradient
evaluations), the iteration count and a convergence flag. The best
parameters are also bound on the `QuMat` instance.

## Backend fixes found while building this

1. **Qiskit state vector after SWAP.** `get_final_state_vector()`
   transpiled at the default optimization level, which elides a SWAP
   into a qubit relabelling that the saved state vector does not
   undo. The fix transpiles at level 1 and caches the simulator
   target (also about 5x faster per call).
2. **Shared parameters on Qiskit.** Using one name in two gates raised
   `CircuitError` on Qiskit only. The fix reuses the existing
   `Parameter` object.

## Alternatives considered

- **Methods on `QuMat`** (for example `qumat.optimize(...)`). Rejected
  for now to keep `QuMat` focused on circuits; a module of functions is
  easier to review and can be wrapped later.
- **Only wrap SciPy.** Simple, but gives no gradient story and no
  Adam, which is the usual choice for variational circuits.
- **Expectation values from shot counts.** Works on real hardware but
  makes gradients noisy and tests flaky. Can be added later as
  `expectation_z(..., shots=N)`.
- **Per-gate shifting** for shared parameters. Needs backend support
  for shifting a single gate; the generalized rule gives exact results
  with the current backend interface.

## Open questions

- Should the optimizer API accept an array-based cost (`f(x)`) in
  addition to the named-dict form?
- Should observables be a small class (`PauliString`) rather than
  `expectation_z` only?
- Should `optimize_parameters` restore the caller's original bindings
  instead of binding the best parameters?
- Is the `count_parameter_occurrences` backend hook acceptable, or
  should parameter bookkeeping move into `QuMat` itself?
