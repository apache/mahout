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
"""Gradients and optimizers for parameterized Qumat circuits.

This module removes the boilerplate of hand-written optimization loops
around :class:`qumat.QuMat` circuits. It provides:

- :func:`canonical_state_vector` and :func:`expectation_z`: exact,
  shot-free expectation values computed from the final state vector,
  with one qubit-ordering convention for every backend.
- :func:`parameter_shift_gradient`: exact gradients via the
  parameter-shift rule for the RX/RY/RZ gates that Qumat exposes.
- :func:`optimize_parameters`: gradient descent and Adam driven by
  parameter-shift gradients, plus a passthrough to ``scipy.optimize``.

A *cost function* has the signature ``cost_fn(qumat, params) -> float``
where ``params`` maps parameter names to values. It is responsible for
binding the values and evaluating the circuit, for example::

    def cost_fn(qumat, params):
        return expectation_z(qumat, params, qubits=0)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .qumat import QuMat

CostFunction = Callable[["QuMat", Mapping[str, float]], float]

# Backends whose state vectors put qubit 0 in the least-significant bit.
# Cirq and Amazon Braket put qubit 0 in the most-significant bit.
_LITTLE_ENDIAN_BACKENDS = frozenset({"qiskit"})

_BUILTIN_METHODS = ("gradient_descent", "adam")
_GRADIENT_METHODS = ("parameter_shift", "finite_difference")
# SciPy methods that accept a ``jac`` (gradient) argument.
_SCIPY_JAC_METHODS = frozenset(
    {"bfgs", "l-bfgs-b", "cg", "slsqp", "tnc", "trust-constr", "newton-cg"}
)
# Cirq returns single-precision (complex64) state vectors, so a smaller
# step would be dominated by round-off error.
_FINITE_DIFFERENCE_STEP = 1e-3


def canonical_state_vector(
    qumat: QuMat, params: Mapping[str, float] | None = None
) -> np.ndarray:
    """Return the final state vector with qubit 0 as the most significant bit.

    Qiskit orders amplitudes little-endian (qubit 0 is the rightmost bit of
    the basis-state index) while Cirq and Amazon Braket are big-endian
    (qubit 0 is the leftmost bit). This helper always returns the big-endian
    ordering, so amplitude ``k`` corresponds to the bit string
    ``format(k, f"0{n}b")`` read as qubit 0, qubit 1, ... on every backend.

    :param qumat: A QuMat instance with a circuit created via
        ``create_empty_circuit(num_qubits)``.
    :param params: Optional parameter values to bind before simulating.
    :returns: Complex numpy array of length ``2**num_qubits``.
    :raises ValueError: If the circuit was created without ``num_qubits``.
    """
    if qumat.num_qubits is None:
        raise ValueError(
            "canonical_state_vector requires a circuit created with "
            "create_empty_circuit(num_qubits=...)"
        )
    if params:
        qumat.bind_parameters(params)
    n = qumat.num_qubits
    state = np.asarray(qumat.get_final_state_vector(), dtype=complex).reshape(-1)
    if state.size != 2**n:
        raise RuntimeError(
            f"expected a state vector of length {2**n}, got {state.size}"
        )
    if qumat.backend_name in _LITTLE_ENDIAN_BACKENDS and n > 1:
        # Axis 0 of the reshaped tensor is the most significant bit, which
        # is qubit n-1 for a little-endian vector. Reversing the axes puts
        # qubit 0 first.
        state = state.reshape([2] * n).transpose().reshape(-1)
    return state


def expectation_z(
    qumat: QuMat,
    params: Mapping[str, float] | None = None,
    qubits: int | Sequence[int] = 0,
) -> float:
    """Exact expectation value of a Pauli-Z product, from the state vector.

    For ``qubits=0`` this is ``<Z_0>``; for ``qubits=[0, 1]`` it is
    ``<Z_0 Z_1>`` (the parity of qubits 0 and 1). The value lies in
    ``[-1, 1]`` and is computed without sampling, so it is deterministic
    and suitable for gradient checks.

    :param qumat: A QuMat instance with an initialized circuit.
    :param params: Optional parameter values to bind before simulating.
    :param qubits: A qubit index or a sequence of qubit indices.
    :returns: The expectation value as a float.
    :raises ValueError: If a qubit index is out of range or no qubits are given.
    """
    targets = [qubits] if isinstance(qubits, int) else list(qubits)
    if not targets:
        raise ValueError("qubits must contain at least one qubit index")
    n = qumat.num_qubits
    for q in targets:
        if not isinstance(q, int) or isinstance(q, bool):
            raise TypeError(f"qubit indices must be integers, got {q!r}")
        if n is not None and not 0 <= q < n:
            raise ValueError(f"qubit index {q} out of range for {n} qubits")
    state = canonical_state_vector(qumat, params)
    n = int(np.log2(state.size))

    probabilities = np.abs(state) ** 2
    indices = np.arange(state.size)
    parity = np.zeros(state.size, dtype=np.int64)
    for q in targets:
        parity ^= (indices >> (n - 1 - q)) & 1
    signs = 1 - 2 * parity
    return float(np.dot(probabilities, signs))


@cache
def _shift_rule(frequency: int) -> tuple[tuple[float, float], ...]:
    """Return ``(shift, coefficient)`` pairs of the parameter-shift rule.

    When a parameter feeds ``R`` Pauli-rotation gates, the cost is a
    trigonometric polynomial in that parameter with integer frequencies
    ``0..R``. Its exact derivative is (Wierichs et al., Quantum 6, 677,
    2022, eq. 16)::

        f'(t) = sum_{mu=1}^{2R} f(t + x_mu) (-1)^(mu-1) / (4R sin^2(x_mu/2))
        x_mu  = (2 mu - 1) pi / (2R)

    For ``R = 1`` this is the familiar two-term rule
    ``f'(t) = [f(t + pi/2) - f(t - pi/2)] / 2``. Shifts are wrapped into
    ``(-pi, pi]``, which is exact because ``f`` is 2*pi-periodic.
    """
    pairs = []
    for mu in range(1, 2 * frequency + 1):
        x = (2 * mu - 1) * math.pi / (2 * frequency)
        coefficient = (-1) ** (mu - 1) / (4 * frequency * math.sin(x / 2) ** 2)
        if x > math.pi:
            x -= 2 * math.pi
        pairs.append((x, coefficient))
    return tuple(pairs)


def _parameter_occurrences(qumat: QuMat) -> dict[str, int]:
    """Count how many gates in the circuit use each named parameter."""
    qumat._ensure_circuit_initialized()
    return qumat.backend_module.count_parameter_occurrences(qumat.circuit)


def _validate_names(qumat: QuMat, names: Sequence[str]) -> None:
    unknown = [name for name in names if name not in qumat.parameters]
    if unknown:
        raise ValueError(
            f"parameters {unknown} not found in circuit. "
            f"available parameters: {list(qumat.parameters)}"
        )


def _as_float_dict(params: Mapping[str, float]) -> dict[str, float]:
    return {name: float(value) for name, value in params.items()}


def parameter_shift_gradient(
    qumat: QuMat,
    cost_fn: CostFunction,
    params: Mapping[str, float],
    param_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Exact gradient of ``cost_fn`` via the parameter-shift rule.

    Every parameterized gate in Qumat (RX, RY, RZ) has the form
    ``exp(-i * theta * P / 2)`` for a Pauli matrix ``P``. For a cost that is
    an expectation value (or any quantity *linear* in the final state, such
    as an outcome probability), the derivative with respect to ``theta`` is
    exactly ``[f(theta + pi/2) - f(theta - pi/2)] / 2``. No step size is
    involved, unlike finite differences.

    If a parameter name is used by several gates, shifting the name moves
    all of those gates at once, and the two-term rule above would be wrong
    (for ``RY(a) RY(a)``, it returns 0 instead of ``-2 sin 2a``). This
    function counts the gates that use each parameter and applies the
    generalized rule with ``2R`` evaluations for ``R`` gates, which is
    exact for any ``R``.

    The rule does **not** hold for costs that are non-linear in the state,
    such as ``(<Z> - y) ** 2``. Differentiate the linear part (``<Z>``) with
    this function and apply the chain rule classically, or use
    ``gradient="finite_difference"`` in :func:`optimize_parameters`.

    :param qumat: A QuMat instance with a parameterized circuit.
    :param cost_fn: Callable ``cost_fn(qumat, params) -> float``.
    :param params: Current values of the parameters being differentiated.
    :param param_names: Subset of ``params`` to differentiate. Defaults to
        all keys of ``params``.
    :returns: Mapping from parameter name to partial derivative.
    :raises ValueError: If a name is not a parameter of the circuit or not
        present in ``params``.
    """
    values = _as_float_dict(params)
    names = list(values) if param_names is None else list(param_names)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"no value given for parameters {missing}")
    _validate_names(qumat, names)

    occurrences = _parameter_occurrences(qumat)
    gradient: dict[str, float] = {}
    saved = dict(qumat.parameters)
    try:
        for name in names:
            frequency = occurrences.get(name, 0)
            if frequency == 0:
                # Registered but used by no gate: the cost cannot depend on it.
                gradient[name] = 0.0
                continue
            total = 0.0
            for shift, coefficient in _shift_rule(frequency):
                shifted = dict(values)
                shifted[name] = values[name] + shift
                total += coefficient * float(cost_fn(qumat, shifted))
            gradient[name] = total
    finally:
        # cost_fn binds shifted values; leave the circuit as we found it.
        qumat.parameters.update(saved)
    return gradient


def finite_difference_gradient(
    qumat: QuMat,
    cost_fn: CostFunction,
    params: Mapping[str, float],
    param_names: Sequence[str] | None = None,
    step: float = _FINITE_DIFFERENCE_STEP,
) -> dict[str, float]:
    """Central finite-difference gradient of ``cost_fn``.

    Works for any differentiable cost, including non-linear ones, but is
    approximate and very sensitive to shot noise. Prefer
    :func:`parameter_shift_gradient` for expectation values.
    """
    values = _as_float_dict(params)
    names = list(values) if param_names is None else list(param_names)
    _validate_names(qumat, names)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"no value given for parameters {missing}")
    gradient: dict[str, float] = {}
    saved = dict(qumat.parameters)
    try:
        for name in names:
            plus = dict(values)
            minus = dict(values)
            plus[name] += step
            minus[name] -= step
            gradient[name] = (
                float(cost_fn(qumat, plus)) - float(cost_fn(qumat, minus))
            ) / (2 * step)
    finally:
        qumat.parameters.update(saved)
    return gradient


@dataclass
class OptimizationResult:
    """Outcome of :func:`optimize_parameters`.

    :ivar params: Best parameter values found (lowest cost seen).
    :ivar cost: Cost at ``params``.
    :ivar history: Cost after each iteration (the first entry is the cost
        at the initial parameters).
    :ivar n_evaluations: Number of ``cost_fn`` calls, including the extra
        calls needed for gradients.
    :ivar n_iterations: Number of optimizer iterations performed.
    :ivar converged: Whether the stopping tolerance was reached (or, for
        SciPy, whether SciPy reported success).
    :ivar method: The optimization method used.
    :ivar message: Human-readable termination reason.
    """

    params: dict[str, float]
    cost: float
    history: list[float] = field(default_factory=list)
    n_evaluations: int = 0
    n_iterations: int = 0
    converged: bool = False
    method: str = ""
    message: str = ""


class _GradientDescent:
    """Plain gradient descent: ``x <- x - lr * g``."""

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return x - self.learning_rate * grad


class _Adam:
    """Adam (Kingma & Ba, 2015) with bias-corrected moment estimates."""

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t = 0

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self._m is None or self._v is None:
            self._m = np.zeros_like(x)
            self._v = np.zeros_like(x)
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad**2
        m_hat = self._m / (1 - self.beta1**self._t)
        v_hat = self._v / (1 - self.beta2**self._t)
        return x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def _make_step_optimizer(
    method: str, learning_rate: float, options: Mapping[str, Any] | None = None
) -> _GradientDescent | _Adam:
    """Build a built-in first-order optimizer (shared with the trainer)."""
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    options = dict(options or {})
    if method == "gradient_descent":
        if options:
            raise ValueError(
                f"unsupported options for gradient_descent: {sorted(options)}"
            )
        return _GradientDescent(learning_rate)
    if method == "adam":
        allowed = {"beta1", "beta2", "epsilon"}
        extra = set(options) - allowed
        if extra:
            raise ValueError(f"unsupported options for adam: {sorted(extra)}")
        return _Adam(learning_rate, **options)
    raise ValueError(
        f"unknown optimizer {method!r}; expected one of {_BUILTIN_METHODS}"
    )


def _gradient_function(
    gradient: str,
) -> Callable[..., dict[str, float]]:
    if gradient == "parameter_shift":
        return parameter_shift_gradient
    if gradient == "finite_difference":
        return finite_difference_gradient
    raise ValueError(
        f"unknown gradient {gradient!r}; expected one of {_GRADIENT_METHODS}"
    )


def optimize_parameters(
    qumat: QuMat,
    cost_fn: CostFunction,
    initial_params: Mapping[str, float],
    method: str = "gradient_descent",
    *,
    learning_rate: float = 0.1,
    max_iter: int = 100,
    tol: float | None = 1e-6,
    gradient: str = "parameter_shift",
    callback: Callable[[int, dict[str, float], float], None] | None = None,
    options: Mapping[str, Any] | None = None,
) -> OptimizationResult:
    """Minimize ``cost_fn`` over the circuit parameters in ``initial_params``.

    :param qumat: A QuMat instance whose circuit contains the parameters.
    :param cost_fn: Callable ``cost_fn(qumat, params) -> float`` to minimize.
    :param initial_params: Starting values. Only these parameters are
        optimized; any other circuit parameter must already be bound.
    :param method: ``"gradient_descent"`` or ``"adam"`` (built-in, using
        ``gradient``), or any ``scipy.optimize.minimize`` method name such
        as ``"COBYLA"``, ``"Nelder-Mead"`` or ``"L-BFGS-B"``. Gradient-based
        SciPy methods receive the ``gradient`` as ``jac``.
    :param learning_rate: Step size for the built-in methods.
    :param max_iter: Maximum number of iterations.
    :param tol: Built-in methods stop when the cost changes by at most
        ``tol`` between iterations; ``None`` always runs ``max_iter``
        iterations. Passed to SciPy as ``tol``.
    :param gradient: ``"parameter_shift"`` (exact for expectation values)
        or ``"finite_difference"`` (for non-linear costs).
    :param callback: Optional ``callback(iteration, params, cost)`` called
        after each iteration.
    :param options: Extra options: ``beta1``/``beta2``/``epsilon`` for Adam,
        or the ``options`` dict forwarded to ``scipy.optimize.minimize``.
    :returns: An :class:`OptimizationResult`. The best parameters are also
        bound on ``qumat`` when this function returns.
    :raises ValueError: On unknown parameters, unbound extra parameters,
        invalid settings or a non-finite cost.
    :raises ImportError: If a SciPy method is requested and SciPy is missing.
    """
    if not initial_params:
        raise ValueError("initial_params must contain at least one parameter")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    names = list(initial_params)
    _validate_names(qumat, names)
    unbound = [
        name
        for name, value in qumat.parameters.items()
        if value is None and name not in initial_params
    ]
    if unbound:
        raise ValueError(
            f"circuit parameters {unbound} are neither in initial_params nor "
            f"bound; bind them with qumat.bind_parameters() first"
        )
    grad_fn = _gradient_function(gradient)

    n_evaluations = 0

    def counted_cost(qm: QuMat, values: Mapping[str, float]) -> float:
        nonlocal n_evaluations
        n_evaluations += 1
        value = float(cost_fn(qm, values))
        if not math.isfinite(value):
            raise ValueError(f"cost_fn returned a non-finite value: {value}")
        return value

    def to_dict(x: np.ndarray) -> dict[str, float]:
        return {name: float(v) for name, v in zip(names, x)}

    x0 = np.array([float(initial_params[name]) for name in names])

    if method in _BUILTIN_METHODS:
        optimizer = _make_step_optimizer(method, learning_rate, options)
        x = x0
        cost = counted_cost(qumat, to_dict(x))
        history = [cost]
        best_x, best_cost = x.copy(), cost
        converged = False
        iteration = 0
        for iteration in range(1, max_iter + 1):
            grad = grad_fn(qumat, counted_cost, to_dict(x), names)
            x = optimizer.step(x, np.array([grad[name] for name in names]))
            new_cost = counted_cost(qumat, to_dict(x))
            history.append(new_cost)
            if new_cost < best_cost:
                best_x, best_cost = x.copy(), new_cost
            if callback is not None:
                callback(iteration, to_dict(x), new_cost)
            if tol is not None and abs(new_cost - cost) <= tol:
                converged = True
                cost = new_cost
                break
            cost = new_cost
        message = "cost change below tol" if converged else "maximum iterations reached"
        best = to_dict(best_x)
        qumat.bind_parameters(best)
        return OptimizationResult(
            params=best,
            cost=best_cost,
            history=history,
            n_evaluations=n_evaluations,
            n_iterations=iteration,
            converged=converged,
            method=method,
            message=message,
        )

    return _optimize_with_scipy(
        qumat,
        counted_cost,
        names,
        x0,
        method,
        max_iter=max_iter,
        tol=tol,
        grad_fn=grad_fn,
        callback=callback,
        options=options,
        evaluations=lambda: n_evaluations,
    )


def _optimize_with_scipy(
    qumat: QuMat,
    cost: CostFunction,
    names: list[str],
    x0: np.ndarray,
    method: str,
    *,
    max_iter: int,
    tol: float | None,
    grad_fn: Callable[..., dict[str, float]],
    callback: Callable[[int, dict[str, float], float], None] | None,
    options: Mapping[str, Any] | None,
    evaluations: Callable[[], int],
) -> OptimizationResult:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - scipy ships with qiskit
        raise ImportError(
            f"method {method!r} requires scipy; install it or use one of "
            f"{_BUILTIN_METHODS}"
        ) from exc

    def to_dict(x: np.ndarray) -> dict[str, float]:
        return {name: float(v) for name, v in zip(names, x)}

    # Remember costs by point so the per-iteration callback does not need
    # to re-run the circuit.
    seen: dict[tuple[float, ...], float] = {}

    def fun(x: np.ndarray) -> float:
        value = cost(qumat, to_dict(x))
        seen[tuple(np.asarray(x, dtype=float))] = value
        return value

    def jac(x: np.ndarray) -> np.ndarray:
        grad = grad_fn(qumat, cost, to_dict(x), names)
        return np.array([grad[name] for name in names])

    history = [fun(x0)]

    def on_iteration(xk: np.ndarray) -> None:
        key = tuple(np.asarray(xk, dtype=float))
        value = seen[key] if key in seen else fun(xk)
        history.append(value)
        if callback is not None:
            callback(len(history) - 1, to_dict(xk), value)

    scipy_options = {"maxiter": max_iter, **dict(options or {})}
    result = minimize(
        fun,
        x0,
        method=method,
        jac=jac if method.lower() in _SCIPY_JAC_METHODS else None,
        tol=tol,
        callback=on_iteration,
        options=scipy_options,
    )
    best = to_dict(np.atleast_1d(result.x))
    best_cost = float(result.fun)
    qumat.bind_parameters(best)
    return OptimizationResult(
        params=best,
        cost=best_cost,
        history=history,
        n_evaluations=evaluations(),
        n_iterations=int(getattr(result, "nit", len(history) - 1)),
        converged=bool(result.success),
        method=method,
        message=str(result.message),
    )
