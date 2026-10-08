#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Quantum (fidelity) kernels built from Qumat circuits.

A quantum kernel compares two classical data points ``x`` and ``y`` by
encoding each into a quantum state with a *feature map* ``U(x)`` and
measuring how similar the two states are:

    K(x, y) = |<phi(x)|phi(y)>|^2,  with |phi(x)> = U(x)|0...0>

The result for a whole dataset is a Gram (kernel) matrix that can be handed
to any classical kernel method, for example scikit-learn's
``SVC(kernel="precomputed")``, kernel ridge regression or kernel PCA.

Three estimation methods are supported:

``"statevector"``
    Simulate each ``|phi(x)>`` exactly and take inner products. Needs only
    one circuit per data point. Exact, but only possible on a simulator.
``"compute_uncompute"``
    Run ``U(y)^dagger U(x)`` on ``n`` qubits and estimate the probability of
    measuring all zeros. One circuit per pair, ``n`` qubits, shot noise.
``"swap_test"``
    Prepare both states side by side on ``2n + 1`` qubits and run a swap
    test; ``K = 2 * P(ancilla = 0) - 1``. One circuit per pair, shot noise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .qumat import QuMat

KERNEL_METHODS: tuple[str, ...] = ("statevector", "compute_uncompute", "swap_test")


@dataclass(frozen=True)
class GateOp:
    """One gate in a feature-map circuit, independent of any backend.

    :param name: Gate name: ``"h"``, ``"rx"``, ``"ry"``, ``"rz"`` or
        ``"cnot"``.
    :param qubits: Qubit indices the gate acts on (control first for CNOT).
    :param angle: Rotation angle in radians for rotation gates, else ``None``.
    """

    name: str
    qubits: tuple[int, ...]
    angle: float | None = None

    def inverse(self) -> GateOp:
        """Return the inverse gate (H and CNOT are self-inverse)."""
        if self.angle is None:
            return self
        return GateOp(self.name, self.qubits, -self.angle)


class FeatureMap:
    """Base class for data-encoding circuits ``U(x)``.

    Subclasses implement :meth:`operations`, returning the gate list for one
    data point. :meth:`apply` then appends it to a :class:`QuMat` circuit,
    optionally shifted to another qubit register or inverted, which is all
    the kernel estimators need.

    :param num_qubits: Number of qubits the feature map acts on.
    """

    def __init__(self, num_qubits: int) -> None:
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError(f"num_qubits must be a positive int, got {num_qubits!r}")
        self.num_qubits = num_qubits

    @property
    def num_features(self) -> int:
        """Number of classical features one data point must have."""
        return self.num_qubits

    def operations(self, x: np.ndarray) -> list[GateOp]:
        """Return the gates that prepare ``|phi(x)>`` from ``|0...0>``."""
        raise NotImplementedError

    def validate(self, x: ArrayLike) -> np.ndarray:
        """Return ``x`` as a 1-D float array, checking length and finiteness."""
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.num_features:
            raise ValueError(
                f"{type(self).__name__} expects {self.num_features} features, "
                f"got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("feature values must be finite")
        return arr

    def apply(
        self,
        qumat: QuMat,
        x: ArrayLike,
        *,
        offset: int = 0,
        inverse: bool = False,
    ) -> None:
        """Append ``U(x)`` (or ``U(x)^dagger``) to ``qumat``'s circuit.

        :param qumat: A QuMat whose circuit has already been created.
        :param x: One data point.
        :param offset: Added to every qubit index, so the map can be placed
            on a register other than ``0..n-1`` (used by the swap test).
        :param inverse: Append the inverse circuit instead.
        """
        ops = self.operations(self.validate(x))
        if inverse:
            ops = [op.inverse() for op in reversed(ops)]
        for op in ops:
            q = [i + offset for i in op.qubits]
            if op.name == "h":
                qumat.apply_hadamard_gate(q[0])
            elif op.name == "cnot":
                qumat.apply_cnot_gate(q[0], q[1])
            elif op.name in ("rx", "ry", "rz") and op.angle is not None:
                # Cast to a built-in float: some backends (Braket) reject
                # numpy scalar types such as np.float32.
                rotate = getattr(qumat, f"apply_{op.name}_gate")
                rotate(q[0], float(op.angle))
            else:
                raise ValueError(f"unsupported gate in feature map: {op!r}")


class AngleFeatureMap(FeatureMap):
    """Product-state angle encoding: ``|phi(x)> = (x) RY(scale * x_k)|0>``.

    One feature per qubit, no entanglement. Its kernel has the closed form

        K(x, y) = prod_k cos^2(scale * (x_k - y_k) / 2)

    which makes it a good correctness check. With ``scale=2`` the state is
    ``(x) (cos x_k |0> + sin x_k |1>)``, the same convention as QDP's
    ``"angle"`` encoding.

    :param num_qubits: Number of qubits (= number of features).
    :param scale: Multiplier applied to every feature before rotation.
    """

    def __init__(self, num_qubits: int, scale: float = 1.0) -> None:
        super().__init__(num_qubits)
        self.scale = float(scale)

    def operations(self, x: np.ndarray) -> list[GateOp]:
        return [
            GateOp("ry", (k,), self.scale * float(x[k])) for k in range(self.num_qubits)
        ]


class ZZFeatureMap(FeatureMap):
    """Second-order Pauli-Z ("ZZ") feature map of Havlicek et al. (2019).

    Each repetition applies ``H`` on every qubit, ``RZ(2 * x_i)`` on qubit
    ``i``, then for every entangled pair ``(i, j)`` the block
    ``CNOT(i, j) RZ(2 * (pi - x_i) * (pi - x_j))[j] CNOT(i, j)``, which is
    ``exp(-i * (pi - x_i)(pi - x_j) Z_i Z_j)``.

    This matches Qiskit's ``zz_feature_map`` (same defaults: ``reps=2``,
    ``entanglement="full"``) up to a global phase, because Qiskit uses the
    phase gate ``P`` where this map uses ``RZ``. Global phase does not change
    ``|<phi(x)|phi(y)>|^2``, so the kernels are identical.

    :param num_qubits: Number of qubits (= number of features), at least 1.
    :param reps: Number of times the encoding block is repeated.
    :param entanglement: ``"full"`` (all pairs ``i < j``), ``"linear"``
        (``(i, i + 1)``) or ``"circular"`` (linear plus ``(n - 1, 0)``).
    """

    _ENTANGLEMENTS = ("full", "linear", "circular")

    def __init__(
        self, num_qubits: int, reps: int = 2, entanglement: str = "full"
    ) -> None:
        super().__init__(num_qubits)
        if not isinstance(reps, int) or reps < 1:
            raise ValueError(f"reps must be a positive int, got {reps!r}")
        if entanglement not in self._ENTANGLEMENTS:
            raise ValueError(
                f"entanglement must be one of {self._ENTANGLEMENTS}, "
                f"got {entanglement!r}"
            )
        self.reps = reps
        self.entanglement = entanglement

    def pairs(self) -> list[tuple[int, int]]:
        """Return the qubit pairs that receive a ZZ interaction."""
        n = self.num_qubits
        if self.entanglement == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        linear = [(i, i + 1) for i in range(n - 1)]
        if self.entanglement == "circular" and n > 2:
            linear.append((n - 1, 0))
        return linear

    def operations(self, x: np.ndarray) -> list[GateOp]:
        ops: list[GateOp] = []
        for _ in range(self.reps):
            for i in range(self.num_qubits):
                ops.append(GateOp("h", (i,)))
            for i in range(self.num_qubits):
                ops.append(GateOp("rz", (i,), 2.0 * float(x[i])))
            for i, j in self.pairs():
                phi = (math.pi - float(x[i])) * (math.pi - float(x[j]))
                ops.append(GateOp("cnot", (i, j)))
                ops.append(GateOp("rz", (j,), 2.0 * phi))
                ops.append(GateOp("cnot", (i, j)))
        return ops


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _config_with_shots(
    backend_config: Mapping[str, Any], shots: int | None
) -> dict[str, Any]:
    """Copy ``backend_config``, overriding ``shots`` when one is given."""
    config = dict(backend_config)
    config["backend_options"] = dict(config.get("backend_options", {}))
    if shots is not None:
        if not isinstance(shots, int) or shots < 1:
            raise ValueError(f"shots must be a positive int, got {shots!r}")
        config["backend_options"]["shots"] = shots
    return config


def _prob_all_zero(results: Any) -> float:
    """Fraction of shots in which every measured qubit read 0.

    The all-zeros outcome looks the same under little- and big-endian
    ordering, so this works for every backend's result format: Qiskit and
    Braket return ``{"00..0": count}``; Cirq returns ``[{0: count}]``.
    """
    if isinstance(results, list):
        results = results[0]
    total = sum(results.values())
    if total == 0:
        return 0.0
    zero = 0
    for state, count in results.items():
        if isinstance(state, str):
            if set(state.replace(" ", "")) <= {"0"}:
                zero += count
        elif int(state) == 0:
            zero += count
    return zero / total


def _statevector(qumat: QuMat, feature_map: FeatureMap, x: np.ndarray) -> np.ndarray:
    qumat.create_empty_circuit(num_qubits=feature_map.num_qubits)
    feature_map.apply(qumat, x)
    return np.asarray(qumat.get_final_state_vector(), dtype=np.complex128)


def _estimate_pair(
    qumat: QuMat,
    feature_map: FeatureMap,
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> float:
    """Shot-based estimate of ``K(x, y)`` with one circuit."""
    n = feature_map.num_qubits
    if method == "compute_uncompute":
        # |<0|U(y)^dagger U(x)|0>|^2 = P(all zeros)
        qumat.create_empty_circuit(num_qubits=n)
        feature_map.apply(qumat, x)
        feature_map.apply(qumat, y, inverse=True)
        return _prob_all_zero(qumat.execute_circuit())

    # swap test: ancilla 0, register A = 1..n, register B = n+1..2n
    qumat.create_empty_circuit(num_qubits=2 * n + 1)
    feature_map.apply(qumat, x, offset=1)
    feature_map.apply(qumat, y, offset=1 + n)
    qumat.apply_hadamard_gate(0)
    for k in range(n):
        qumat.apply_cswap_gate(0, 1 + k, 1 + n + k)
    qumat.apply_hadamard_gate(0)
    # The backend helper knows each backend's bit ordering.
    p0 = qumat.calculate_prob_zero(qumat.execute_circuit(), 0)
    # P(0) = (1 + K) / 2. Clip (do not fold) values pushed outside [0, 1]
    # by shot noise.
    return float(min(1.0, max(0.0, 2.0 * p0 - 1.0)))


def _check_method(method: str) -> None:
    if method not in KERNEL_METHODS:
        raise ValueError(f"method must be one of {KERNEL_METHODS}, got {method!r}")


def _as_2d(data: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (samples x features), got {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def kernel_from_states(
    states_x: ArrayLike,
    states_y: ArrayLike | None = None,
) -> np.ndarray:
    """Fidelity kernel matrix from already-prepared state vectors.

    ``K[i, j] = |<a_i|b_j>|^2`` for rows ``a_i`` of ``states_x`` and ``b_j``
    of ``states_y`` (defaults to ``states_x``). Useful for states produced
    elsewhere, e.g. by QDP's GPU encoders. All states must use the same
    qubit-ordering convention; mixing little- and big-endian vectors gives
    wrong answers for entangled or asymmetric states.

    :returns: Real array of shape ``(len(states_x), len(states_y))``.
    """
    a = np.asarray(states_x, dtype=np.complex128)
    b = a if states_y is None else np.asarray(states_y, dtype=np.complex128)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            f"state arrays must be 2-D with equal width, got {a.shape} and {b.shape}"
        )
    gram = np.abs(a.conj() @ b.T) ** 2
    gram = np.clip(gram, 0.0, 1.0)
    if states_y is None:
        gram = (gram + gram.T) / 2.0
        np.fill_diagonal(gram, 1.0)
    return gram


def fidelity_kernel(
    x1: ArrayLike,
    x2: ArrayLike,
    feature_map: FeatureMap,
    backend_config: Mapping[str, Any],
    *,
    method: str = "statevector",
    shots: int | None = None,
) -> float:
    """Compute one kernel value ``K(x1, x2) = |<phi(x1)|phi(x2)>|^2``.

    :param x1: First data point (length ``feature_map.num_features``).
    :param x2: Second data point.
    :param feature_map: The encoding circuit, e.g. :class:`AngleFeatureMap`.
    :param backend_config: A normal QuMat backend configuration.
    :param method: ``"statevector"`` (exact, simulator only),
        ``"compute_uncompute"`` or ``"swap_test"`` (shot-based).
    :param shots: Overrides ``backend_options["shots"]`` for the shot-based
        methods. Ignored by ``"statevector"``.
    :returns: Kernel value in ``[0, 1]``.
    """
    _check_method(method)
    x1 = feature_map.validate(x1)
    x2 = feature_map.validate(x2)
    qumat = QuMat(_config_with_shots(backend_config, shots))
    if method == "statevector":
        a = _statevector(qumat, feature_map, x1)
        b = _statevector(qumat, feature_map, x2)
        return float(min(1.0, abs(np.vdot(a, b)) ** 2))
    return _estimate_pair(qumat, feature_map, x1, x2, method)


def kernel_matrix(
    X: ArrayLike,
    Y: ArrayLike | None = None,
    *,
    feature_map: FeatureMap,
    backend_config: Mapping[str, Any],
    method: str = "statevector",
    shots: int | None = None,
) -> np.ndarray:
    """Compute the quantum kernel (Gram) matrix between two datasets.

    ``K[i, j] = |<phi(X[i])|phi(Y[j])>|^2``. Use ``Y=None`` for the training
    Gram matrix: then only the upper triangle is estimated (``N(N-1)/2``
    circuits instead of ``N^2``), the result is mirrored so it is exactly
    symmetric, and the diagonal is set to 1 (``K(x, x) = 1`` for any pure
    state). For prediction pass the test set as ``X`` and the training set
    as ``Y``; the result has shape ``(len(X), len(Y))``, as expected by
    ``SVC(kernel="precomputed").predict``.

    The ``"statevector"`` method simulates each data point once (``N``
    circuits) and computes all inner products classically. Overlaps do not
    depend on the qubit ordering convention as long as all states come from
    the same backend, which is always the case here.

    Shot-based Gram matrices are symmetric but not guaranteed to be
    positive semi-definite; see the design notes in
    ``docs/advanced/quantum-kernels.md``.

    :param X: Array of shape ``(n_samples_x, n_features)``.
    :param Y: Optional array of shape ``(n_samples_y, n_features)``.
    :param feature_map: The encoding circuit.
    :param backend_config: A normal QuMat backend configuration.
    :param method: One of :data:`KERNEL_METHODS`.
    :param shots: Shots per circuit for the shot-based methods.
    :returns: Real array of shape ``(n_samples_x, n_samples_y)``.
    """
    _check_method(method)
    xs = _as_2d(X, "X")
    ys = None if Y is None else _as_2d(Y, "Y")
    for row in xs if ys is None else np.vstack([xs, ys]):
        feature_map.validate(row)
    qumat = QuMat(_config_with_shots(backend_config, shots))

    if method == "statevector":
        sx = np.array([_statevector(qumat, feature_map, x) for x in xs])
        if ys is None:
            return kernel_from_states(sx)
        sy = np.array([_statevector(qumat, feature_map, y) for y in ys])
        return kernel_from_states(sx, sy)

    if ys is None:
        n = len(xs)
        gram = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                value = _estimate_pair(qumat, feature_map, xs[i], xs[j], method)
                gram[i, j] = gram[j, i] = value
        return gram

    gram = np.empty((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            gram[i, j] = _estimate_pair(qumat, feature_map, x, y, method)
    return gram


__all__ = [
    "KERNEL_METHODS",
    "AngleFeatureMap",
    "FeatureMap",
    "GateOp",
    "ZZFeatureMap",
    "fidelity_kernel",
    "kernel_from_states",
    "kernel_matrix",
]
