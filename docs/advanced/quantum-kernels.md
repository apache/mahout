---
title: Quantum Kernel Methods - Design Notes
sidebar_label: Quantum Kernel Methods
---

# Quantum Kernel Methods: Design Notes

This page records the outcome of the quantum kernel research spike
([#456](https://github.com/apache/mahout/issues/456)) and the scope
assessment for downstream kernel calculations
([#458](https://github.com/apache/mahout/issues/458)). It is written as an
architecture decision record: the questions the spike asked, the answers,
and what Qumat implements as a result (`qumat/kernels.py`).

**Status:** proposed.

---

## 1. Context

A *kernel method* is a classical learning algorithm (SVM, kernel ridge
regression, Gaussian processes, kernel PCA, ...) that only ever looks at the
data through a similarity function $k(x, y)$. If you can compute $k$ for
every pair of points, you can train the model without knowing anything
else about how $k$ was obtained.

A *quantum kernel* uses a quantum circuit to compute that similarity. Each
data point $x$ is encoded into a quantum state by a parameterized circuit
$U(x)$, the **feature map**:

$$
\vert\phi(x)\rangle = U(x)\,\vert 0\cdots 0\rangle
$$

and the kernel is the **fidelity** (squared overlap) of two such states:

$$
k(x, y) = \bigl\vert \langle \phi(x) \vert \phi(y) \rangle \bigr\vert^2
$$

The hope is that, for a well-chosen $U$, this similarity is useful for
learning and hard to compute classically (Havlicek et al. 2019, *Supervised
learning with quantum-enhanced feature spaces*; Schuld & Killoran 2019,
*Quantum machine learning in feature Hilbert spaces*).

The quantum computer's job is therefore narrow: **estimate kernel values**.
Everything else is classical.

---

## 2. Spike questions and answers

### 2.1 What is the output?

A real **kernel (Gram) matrix**.

- For training data $X = (x_1, \dots, x_N)$: an $N \times N$ matrix with
  $K_{ij} = k(x_i, x_j)$. It is symmetric, has ones on the diagonal
  ($k(x, x) = 1$ for any pure state), entries in $[0, 1]$, and is
  positive semi-definite (PSD) when computed exactly, because it is a
  matrix of inner products.
- For prediction: an $M \times N$ matrix $K_{ij} = k(x^{\text{test}}_i,
  x^{\text{train}}_j)$, rows are test points, columns are training points.

These are plain NumPy arrays. That is the contract with the downstream
algorithms; no quantum objects leak out.

### 2.2 How does the kernel feed classical algorithms?

Every mainstream kernel learner accepts a precomputed Gram matrix:

```python
from sklearn.svm import SVC

K_train = kernel_matrix(X_train, feature_map=fm, backend_config=cfg)
K_test = kernel_matrix(X_test, X_train, feature_map=fm, backend_config=cfg)

clf = SVC(kernel="precomputed").fit(K_train, y_train)
y_pred = clf.predict(K_test)
```

The same two matrices work with `KernelRidge(kernel="precomputed")`,
`KernelPCA(kernel="precomputed")`, `SpectralClustering(affinity="precomputed")`
and Gaussian-process code that takes a covariance matrix. Kernel ridge is
small enough to write by hand, solving $(K + \lambda I)\alpha = y$ and
predicting $K_{\text{test}}\,\alpha$. The example
`examples/qumat/quantum_kernel_classification.py` does exactly that and
uses scikit-learn only if it happens to be installed. Qumat does **not**
depend on scikit-learn.

### 2.3 Which feature maps and gates are needed?

Two feature maps cover the standard literature and the existing gate set:

**Angle (product) map** — one $R_Y$ per feature:

$$
U(x) = \bigotimes_{k} R_Y(s \cdot x_k)
$$

No entanglement, so it is classically easy, but its kernel has a closed
form, which makes it the reference for testing:

$$
k(x, y) = \prod_k \cos^2\!\left(\frac{s\,(x_k - y_k)}{2}\right)
$$

With scale $s = 2$ the state is $\bigotimes_k (\cos x_k \vert 0\rangle +
\sin x_k \vert 1\rangle)$, the same convention as QDP's `"angle"` encoding.

**ZZ map** (Havlicek et al.) — per repetition: $H$ on every qubit,
$R_Z(2x_i)$ on each qubit, then for each entangled pair $(i, j)$

$$
\text{CNOT}_{ij}\; R_Z\bigl(2(\pi - x_i)(\pi - x_j)\bigr)_j\; \text{CNOT}_{ij}
= e^{-i (\pi - x_i)(\pi - x_j) Z_i Z_j}
$$

This is the IQP-style map that motivates quantum kernels. Our version
equals Qiskit's `zz_feature_map` up to a global phase (Qiskit uses the
phase gate $P$ where we use $R_Z$), so its kernel values are identical;
the test suite checks this to $10^{-9}$.

**Gates needed:** $H$, $R_Y$, $R_Z$, CNOT for the feature maps; CSWAP for
the swap test. All already exist in Qumat on all three backends. *No new
gates are required.* Optional extras for richer maps (not needed now):
$R_{ZZ}$ as a native gate (saves two CNOTs per pair on some hardware) and
a parameterized $U$ gate.

### 2.4 How is the overlap estimated?

Three methods are implemented. They give the same number up to shot noise.

| Method | Qubits | Circuits for N×N | Exact? |
|---|---|---|---|
| `statevector` | $n$ | $N$ | yes (simulator only) |
| `compute_uncompute` | $n$ | $N(N-1)/2$ | no, shot noise |
| `swap_test` | $2n+1$ | $N(N-1)/2$ | no, shot noise |

**Statevector.** Simulate each $\vert\phi(x_i)\rangle$ once, stack the
vectors as the rows of a matrix $S$, and compute $K = \lvert \bar{S} S^{T} \rvert^2$
(element-wise square) classically. This is how
you prototype and test; it is impossible on real hardware and the state
vectors cost $2^n \times 16$ bytes each.

**Compute-uncompute** (a.k.a. the inversion test). Run
$U(y)^\dagger U(x)$ and measure. Because
$\langle 0 \vert U(y)^\dagger U(x) \vert 0 \rangle = \langle\phi(y)\vert\phi(x)\rangle$,

$$
k(x, y) = P(\text{all qubits read } 0)
$$

Same width as the feature map, double the depth. The inverse is cheap to
build: reverse the gate list and negate rotation angles ($H$ and CNOT are
self-inverse). This is the method used in most hardware experiments.

**Swap test.** Prepare $\vert\phi(x)\rangle$ on register A and
$\vert\phi(y)\rangle$ on register B, then $H$ on an ancilla, one CSWAP per
qubit pair (ancilla controls, $A_k \leftrightarrow B_k$), $H$ again:

$$
P(\text{ancilla} = 0) = \frac{1 + k(x, y)}{2}
\quad\Rightarrow\quad
k = 2P_0 - 1
$$

Needs $2n + 1$ qubits and $n$ three-qubit gates, so it is the most
expensive on hardware. It does not need $U^\dagger$, which matters only
if the states come from a black box. Qumat's existing `swap_test()` uses a
single CSWAP, so it only works for one-qubit registers; `qumat.kernels`
applies one CSWAP per qubit.

**Shot noise.** With $S$ shots the standard error of a probability is at
most $0.5/\sqrt{S}$, so compute-uncompute gives roughly $\pm 1/(2\sqrt{S})$
and the swap test, which doubles $P_0$, roughly $\pm 1/\sqrt{S}$.
1000 shots give about $\pm 0.03$ on a swap-test kernel entry.

### 2.5 Cost

- Exact: $N$ state simulations plus an $O(N^2 2^n)$ matrix product.
- Shot-based training matrix: $N(N-1)/2$ circuits (symmetry plus unit
  diagonal halve the $N^2$ work). Test matrix: $MN$ circuits; no symmetry.
- Each circuit costs `shots` executions. For $N = 1000$, that is about
  500 000 circuits for training alone, which is why quantum kernel papers
  use small datasets, and why batching and cloud job submission are a
  separate concern (see §4).

### 2.6 Qubit ordering

Qiskit is little-endian (qubit 0 is the lowest bit of the state index);
Cirq and Braket are big-endian (qubit 0 is the highest bit). QDP's angle
kernel is little-endian, like Qiskit. The same physical state therefore
has *permuted* amplitudes on different backends.

This does not affect the kernel, as long as both states in an overlap
come from the same convention: a fixed permutation $P$ satisfies
$\langle P a, P b \rangle = \langle a, b \rangle$. `kernel_matrix` always
builds every state with one backend, so the result is ordering-invariant;
the tests check that Qiskit, Cirq and Braket agree to $10^{-6}$ (the
tolerance is set by Cirq's `complex64` state vectors). The two places
where ordering *does* matter are:

- reading a single qubit from shot results (the swap-test ancilla). This
  goes through each backend's `calculate_prob_zero`, which already handles
  the convention;
- mixing state vectors from different sources (e.g. QDP and Cirq) in
  `kernel_from_states`, which callers must avoid.

The all-zeros outcome used by compute-uncompute looks the same in every
convention, so it needs no special handling.

### 2.7 About `measure_overlap` and `max(p0, 1 - p0)`

`QuMat.measure_overlap` computes $2\max(P_0, 1 - P_0) - 1$. In an ideal
swap test $P_0 = (1 + F)/2 \ge 1/2$ always, so the fold should never be
needed. Its docstring attributes $P_0 \approx 0$ for two $\vert 1\rangle$
states to "global phase effects", but a global phase cannot change any
measurement probability.

What actually happens: the hand-written `calculate_prob_zero` helper in
`testing/qumat/test_swap_test.py` always reads the *last* bit (string
results) or the *lowest* bit (Cirq integers). That is the ancilla for
Qiskit, but qubit 2 for Cirq and Braket. With $\vert 1 1\rangle$ on qubits
1 and 2, it reads qubit 2's value, 1, and reports $P_0 = 0$. The backend
`calculate_prob_zero` that `measure_overlap` really calls gives
$P_0 = 1.000$ on all three backends for that circuit, and the whole
`test_overlap_measurement.py` suite still passes with the fold removed.

The fold is harmful in two ways:

1. It **hides bugs**. If the ancilla bit is read wrongly or a circuit is
   wired backwards ($P_0 \approx 0$), the method reports fidelity
   $\approx 1$ instead of a value that is obviously wrong.
2. It **biases small fidelities upward** under shot noise. For $F = 0$,
   $P_0$ scatters around $1/2$; folding returns $\vert 2P_0 - 1\vert$, about
   $0.8/\sqrt{S}$ on average, about twice the bias of plain clipping.

Recommendation (a follow-up, not part of this change): replace the fold
with `max(0.0, 2 * p0 - 1)`, fix the docstring, and fix the test helper to
use the backend's `calculate_prob_zero`. `qumat.kernels` already clips
instead of folding.

---

## 3. Sub-issue status (verified against the code at commit `2d94f8f`)

| Issue | Topic | Status in code |
|---|---|---|
| #468 | Parameter object | **Partial.** Parameters are plain strings tracked in `QuMat.parameters`; there is no `Parameter` class. Enough for kernels. |
| #464–#467 | Parameterize RX / RY / RZ / U | **RX, RY, RZ done** (string angles on all three backends; `test_rotation_gates.py`, `test_parameter_binding.py`). **U not done**: `apply_u_gate(0, "t", 0, 0)` raises on Qiskit, Cirq and Braket. Not needed for the maps here. |
| #469 | Bind values at execution | **Done.** `execute_circuit(parameter_values=...)` and `bind_parameters()`; tests in `test_parameter_binding.py`. |
| #476 | Overlap measurements | **Done**, with the caveat in §2.7 (`measure_overlap` on single qubits). |
| #477 | Swap test | **Done** for one-qubit registers (`QuMat.swap_test`, `test_swap_test.py`). Multi-qubit version in `qumat.kernels`. |
| #457 | Data encoder design | **Open.** QDP has GPU encoders (amplitude, angle, basis, phase, IQP) that output state vectors; Qumat had no circuit-level encoder. `FeatureMap` here is a first, kernel-oriented answer. |
| #478 | Post-processing | **Started here**: Gram assembly, symmetry, unit diagonal, clipping. PSD repair and error mitigation remain (§4). |
| #458 | Downstream kernel scope | **This document** (§4). |

(The issue text lists the gate sub-issues as "#464/#466/#465/#467" for
RX/RY/RZ/U, so the exact number for each gate is not certain.)

---

## 4. Scope for #458

### In scope (implemented in `qumat.kernels`)

- `FeatureMap` base class with a backend-independent gate list, used both
  forward and inverted, and placeable on any qubit offset.
- `AngleFeatureMap` and `ZZFeatureMap` (full, linear, circular
  entanglement; `reps`).
- `fidelity_kernel(x1, x2, feature_map, backend_config, method=..., shots=...)`.
- `kernel_matrix(X, Y=None, ...)` with symmetry and unit diagonal when
  `Y is None`.
- `kernel_from_states(...)` to build a kernel from precomputed state
  vectors (e.g. from QDP).
- Three estimators: `statevector`, `compute_uncompute`, `swap_test`.
- NumPy output that drops into `SVC(kernel="precomputed")` and friends.

### Out of scope (candidate follow-ups)

- **PSD repair** for shot-based matrices. Shot noise can make
  $K$ slightly indefinite. Options: clip negative eigenvalues, or the
  "nearest PSD matrix" projection. SVC usually tolerates small violations.
- **Error mitigation** on hardware (readout correction, zero-noise
  extrapolation).
- **Trainable kernels** and kernel-target alignment (optimizing feature
  map parameters). This needs the optimizer work in #749 / #1409.
- **Projected quantum kernels** (Huang et al. 2021), which compare
  reduced density matrices instead of full overlaps and suffer less from
  exponential concentration.
- **Batching and job submission**: build all $N(N-1)/2$ circuits and send
  them as one batch (important for cloud backends such as Braket SV1,
  where per-task overhead dominates). The current loop runs circuits one
  by one.
- **Circuit reuse via parameter binding**: build the feature map once with
  symbolic parameters and bind per data point. Qumat parameters are plain
  names, not expressions, so the ZZ product terms would need one parameter
  per pair. Rebuilding is simple and cheap at the sizes simulators handle.
- **QDP GPU path**: encode a whole batch with QDP (`"angle"`, `"iqp"`) and
  call `kernel_from_states`; the kernel becomes one GPU matrix product.
  Needs a GPU to test.
- **Wrapping scikit-learn** (`QuantumKernel` estimator object). Easy to
  add later; kept out to avoid a hard dependency.

### Known limits

- **Exponential concentration.** As qubits or depth grow, off-diagonal
  values of expressive maps shrink toward 0 and $K \to I$, which learns
  nothing. Keep feature maps shallow and scale features (the `scale`
  argument of `AngleFeatureMap`; `reps=1` for `ZZFeatureMap`).
- **No free advantage.** The angle map is classically simulable; the ZZ
  map is believed hard to simulate but is not automatically a good kernel
  for a given dataset. In the example script the angle map wins on two
  Gaussian clusters.

---

## 5. Decision

1. A quantum kernel in Qumat is a function from two data arrays to a NumPy
   kernel matrix, independent of the backend.
2. Feature maps are described as backend-independent gate lists and
   rendered with existing Qumat gates; no new gates are needed.
3. Statevector estimation is the default (exact, fast to test);
   compute-uncompute is the recommended shot-based method; the swap test
   is kept for completeness and for black-box state preparation.
4. Downstream learning stays classical and outside Qumat; the
   precomputed-kernel interface is the integration point.

## References

- V. Havlicek et al., *Supervised learning with quantum-enhanced feature
  spaces*, Nature 567 (2019). arXiv:1804.11326
- M. Schuld, N. Killoran, *Quantum machine learning in feature Hilbert
  spaces*, PRL 122 (2019). arXiv:1803.07128
- H.-Y. Huang et al., *Power of data in quantum machine learning*,
  Nature Communications 12 (2021). arXiv:2011.01938
- S. Thanasilp et al., *Exponential concentration in quantum kernel
  methods*, Nature Communications 15 (2024). arXiv:2208.11060
- [Quantum Kernel Estimation With Neutral Atoms](../learning/papers/Quantum-Kernel-Estimation-With-Neutral-Atoms-For-Supervised-Classification)
  (summary in these docs)
