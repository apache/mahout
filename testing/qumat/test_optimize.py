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
"""Tests for qumat.optimize (parameter-shift gradients and optimizers)."""

import math

import pytest

from qumat import (
    OptimizationResult,
    QuMat,
    expectation_z,
    optimize_parameters,
    parameter_shift_gradient,
)
from qumat.optimize import canonical_state_vector, finite_difference_gradient

from ..utils import TESTING_BACKENDS

# Cirq simulates in single precision (complex64); allow for that.
ATOL = 1e-5

# Fast, exact backends for the optimizer tests.
OPTIMIZER_BACKENDS = ["qiskit", "cirq"]


def make_qumat(backend_name, num_qubits):
    configs = {
        "qiskit": {"simulator_type": "statevector_simulator", "shots": 1},
        "cirq": {"simulator_type": "default", "shots": 1},
        "amazon_braket": {"simulator_type": "local", "shots": 1},
    }
    qumat = QuMat(
        {"backend_name": backend_name, "backend_options": configs[backend_name]}
    )
    qumat.create_empty_circuit(num_qubits=num_qubits)
    return qumat


def z0_cost(qumat, params):
    return expectation_z(qumat, params, qubits=0)


class TestExpectationZ:
    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_qubit_ordering_is_consistent(self, backend_name):
        """X on qubit 0 must flip <Z_0> only, whatever the backend's ordering."""
        qumat = make_qumat(backend_name, 3)
        qumat.apply_pauli_x_gate(0)

        state = canonical_state_vector(qumat)
        # Canonical order is big-endian: |q0 q1 q2> = |100> is index 4.
        assert abs(state[4]) == pytest.approx(1.0, abs=ATOL)

        assert expectation_z(qumat, qubits=0) == pytest.approx(-1.0, abs=ATOL)
        assert expectation_z(qumat, qubits=1) == pytest.approx(1.0, abs=ATOL)
        assert expectation_z(qumat, qubits=2) == pytest.approx(1.0, abs=ATOL)
        assert expectation_z(qumat, qubits=[0, 1]) == pytest.approx(-1.0, abs=ATOL)

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_bell_state_parity(self, backend_name):
        qumat = make_qumat(backend_name, 2)
        qumat.apply_hadamard_gate(0)
        qumat.apply_cnot_gate(0, 1)

        assert expectation_z(qumat, qubits=0) == pytest.approx(0.0, abs=ATOL)
        assert expectation_z(qumat, qubits=[0, 1]) == pytest.approx(1.0, abs=ATOL)

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_binds_params(self, backend_name):
        qumat = make_qumat(backend_name, 1)
        qumat.apply_ry_gate(0, "theta")
        value = expectation_z(qumat, {"theta": 0.7})
        assert value == pytest.approx(math.cos(0.7), abs=ATOL)
        assert qumat.parameters["theta"] == 0.7

    def test_invalid_qubits(self):
        qumat = make_qumat("cirq", 2)
        with pytest.raises(ValueError, match="out of range"):
            expectation_z(qumat, qubits=2)
        with pytest.raises(ValueError, match="at least one"):
            expectation_z(qumat, qubits=[])
        with pytest.raises(TypeError, match="integers"):
            expectation_z(qumat, qubits=[0.5])  # type: ignore[invalid-argument-type]

    def test_requires_num_qubits(self):
        qumat = QuMat({"backend_name": "cirq", "backend_options": {}})
        qumat.create_empty_circuit()
        with pytest.raises(ValueError, match="num_qubits"):
            canonical_state_vector(qumat)


class TestParameterShiftGradient:
    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    @pytest.mark.parametrize("theta", [0.0, 0.4, math.pi / 2, 2.5, -1.3])
    def test_ry_matches_analytic(self, backend_name, theta):
        """For RY(theta)|0>, <Z> = cos(theta) so d<Z>/dtheta = -sin(theta)."""
        qumat = make_qumat(backend_name, 1)
        qumat.apply_ry_gate(0, "theta")

        grad = parameter_shift_gradient(qumat, z0_cost, {"theta": theta})

        assert grad["theta"] == pytest.approx(-math.sin(theta), abs=ATOL)

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_rx_and_rz_match_analytic(self, backend_name):
        """RX(a)|0> gives <Z> = cos(a); H RZ(c) H acts like RX(c)."""
        qumat = make_qumat(backend_name, 2)
        qumat.apply_rx_gate(0, "alpha")
        qumat.apply_hadamard_gate(1)
        qumat.apply_rz_gate(1, "gamma")
        qumat.apply_hadamard_gate(1)

        def cost(qm, params):
            return expectation_z(qm, params, 0) + 2 * expectation_z(qm, params, 1)

        grad = parameter_shift_gradient(qumat, cost, {"alpha": 0.3, "gamma": 1.9})

        assert grad["alpha"] == pytest.approx(-math.sin(0.3), abs=ATOL)
        assert grad["gamma"] == pytest.approx(-2 * math.sin(1.9), abs=ATOL)

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_matches_finite_differences(self, backend_name):
        """Entangling circuit using all three rotation gates."""
        qumat = make_qumat(backend_name, 2)
        qumat.apply_ry_gate(0, "t0")
        qumat.apply_rx_gate(1, "t1")
        qumat.apply_cnot_gate(0, 1)
        qumat.apply_rz_gate(1, "t2")
        qumat.apply_hadamard_gate(1)
        qumat.apply_ry_gate(1, "t3")

        def cost(qm, params):
            return expectation_z(qm, params, [0, 1]) - 0.5 * expectation_z(
                qm, params, 1
            )

        params = {"t0": 0.3, "t1": -0.8, "t2": 1.2, "t3": 0.5}
        shift = parameter_shift_gradient(qumat, cost, params)
        finite = finite_difference_gradient(qumat, cost, params)

        for name in params:
            assert shift[name] == pytest.approx(finite[name], abs=1e-3), name
        # Restores the caller's binding state.
        assert all(qumat.parameters[name] is None for name in params)

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_shared_parameter(self, backend_name):
        """A parameter used by two gates needs the generalized rule.

        RY(a) RY(a) = RY(2a): <Z> = cos(2a), derivative -2 sin(2a). The plain
        two-term rule would give [cos(2a + pi) - cos(2a - pi)] / 2 = 0.
        """
        qumat = make_qumat(backend_name, 1)
        qumat.apply_ry_gate(0, "a")
        qumat.apply_ry_gate(0, "a")

        grad = parameter_shift_gradient(qumat, z0_cost, {"a": 0.4})

        assert grad["a"] == pytest.approx(-2 * math.sin(0.8), abs=ATOL)

    def test_shared_parameter_across_gate_types(self):
        qumat = make_qumat("cirq", 2)
        qumat.apply_ry_gate(0, "a")
        qumat.apply_rx_gate(1, "a")
        qumat.apply_rz_gate(1, "a")
        qumat.apply_cnot_gate(0, 1)
        qumat.apply_ry_gate(1, "a")

        def cost(qm, params):
            return expectation_z(qm, params, [0, 1])

        shift = parameter_shift_gradient(qumat, cost, {"a": 0.9})
        finite = finite_difference_gradient(qumat, cost, {"a": 0.9})
        assert shift["a"] == pytest.approx(finite["a"], abs=1e-3)

    def test_param_names_subset(self):
        qumat = make_qumat("cirq", 1)
        qumat.apply_ry_gate(0, "x")
        qumat.apply_rx_gate(0, "y")
        grad = parameter_shift_gradient(
            qumat, z0_cost, {"x": 0.1, "y": 0.2}, param_names=["y"]
        )
        assert list(grad) == ["y"]

    def test_errors(self):
        qumat = make_qumat("cirq", 1)
        qumat.apply_ry_gate(0, "theta")
        with pytest.raises(ValueError, match="not found in circuit"):
            parameter_shift_gradient(qumat, z0_cost, {"theta": 0.1, "phi": 0.2})
        with pytest.raises(ValueError, match="no value given"):
            parameter_shift_gradient(qumat, z0_cost, {}, param_names=["theta"])


class TestOptimizeParameters:
    @pytest.mark.parametrize("backend_name", OPTIMIZER_BACKENDS)
    @pytest.mark.parametrize(
        ("method", "learning_rate"), [("gradient_descent", 0.4), ("adam", 0.2)]
    )
    def test_builtin_methods_converge(self, backend_name, method, learning_rate):
        """Minimize <Z_0> + <Z_1>; the minimum is -2 (both qubits in |1>)."""
        qumat = make_qumat(backend_name, 2)
        qumat.apply_ry_gate(0, "a")
        qumat.apply_cnot_gate(0, 1)
        qumat.apply_ry_gate(1, "c")

        def cost(qm, params):
            return expectation_z(qm, params, 0) + expectation_z(qm, params, 1)

        result = optimize_parameters(
            qumat,
            cost,
            {"a": 0.3, "c": 0.2},
            method=method,
            learning_rate=learning_rate,
            max_iter=80,
            tol=1e-7,
        )

        assert isinstance(result, OptimizationResult)
        assert result.cost == pytest.approx(-2.0, abs=1e-3)
        assert result.history[0] > result.history[-1]
        assert result.history[0] == pytest.approx(
            math.cos(0.3) + math.cos(0.3) * math.cos(0.2), abs=ATOL
        )
        # Best params are bound on the circuit when optimization ends.
        assert qumat.parameters == result.params
        assert expectation_z(qumat) == pytest.approx(-1.0, abs=1e-3)

    def test_evaluation_count_gradient_descent(self):
        """One initial call, then per iteration 2 shift calls per parameter + 1."""
        qumat = make_qumat("cirq", 1)
        qumat.apply_ry_gate(0, "theta")

        seen = []
        result = optimize_parameters(
            qumat,
            z0_cost,
            {"theta": 0.5},
            learning_rate=0.1,
            max_iter=5,
            tol=None,
            callback=lambda i, params, cost: seen.append((i, cost)),
        )

        assert result.n_iterations == 5
        assert result.n_evaluations == 1 + 5 * (2 + 1)
        assert len(result.history) == 6
        assert [i for i, _ in seen] == [1, 2, 3, 4, 5]
        assert not result.converged
        # Hand-rolled gradient descent on cos(theta) for comparison.
        theta = 0.5
        for _ in range(5):
            theta += 0.1 * math.sin(theta)
        assert result.params["theta"] == pytest.approx(theta, abs=ATOL)

    def test_finite_difference_for_nonlinear_cost(self):
        """Squared error is not linear in the state: use finite differences."""
        qumat = make_qumat("cirq", 1)
        qumat.apply_ry_gate(0, "theta")
        target = 0.25

        def cost(qm, params):
            return (expectation_z(qm, params) - target) ** 2

        result = optimize_parameters(
            qumat,
            cost,
            {"theta": 0.1},
            method="adam",
            learning_rate=0.1,
            max_iter=200,
            gradient="finite_difference",
        )
        assert result.cost < 1e-4
        assert math.cos(result.params["theta"]) == pytest.approx(target, abs=0.02)

    @pytest.mark.parametrize("method", ["COBYLA", "Nelder-Mead", "L-BFGS-B"])
    def test_scipy_methods(self, method):
        pytest.importorskip("scipy")
        qumat = make_qumat("cirq", 2)
        qumat.apply_ry_gate(0, "a")
        qumat.apply_rx_gate(1, "c")

        def cost(qm, params):
            return expectation_z(qm, params, 0) + expectation_z(qm, params, 1)

        result = optimize_parameters(
            qumat, cost, {"a": 0.4, "c": -0.3}, method=method, max_iter=200
        )

        assert result.method == method
        assert result.cost == pytest.approx(-2.0, abs=1e-3)
        assert result.n_evaluations > 0
        assert len(result.history) >= 2
        assert qumat.parameters == result.params

    def test_optimizes_subset_with_other_params_bound(self):
        qumat = make_qumat("cirq", 1)
        qumat.apply_ry_gate(0, "fixed")
        qumat.apply_ry_gate(0, "free")
        qumat.bind_parameters({"fixed": 0.5})

        result = optimize_parameters(
            qumat, z0_cost, {"free": 0.0}, method="L-BFGS-B", max_iter=50
        )

        assert result.params["free"] + 0.5 == pytest.approx(math.pi, abs=1e-3)
        assert qumat.parameters["fixed"] == 0.5

    def test_error_paths(self):
        qumat = make_qumat("cirq", 1)
        qumat.apply_ry_gate(0, "theta")
        qumat.apply_ry_gate(0, "phi")

        with pytest.raises(ValueError, match="at least one parameter"):
            optimize_parameters(qumat, z0_cost, {})
        with pytest.raises(ValueError, match="not found in circuit"):
            optimize_parameters(qumat, z0_cost, {"theta": 0, "nope": 0, "phi": 0})
        with pytest.raises(ValueError, match="neither in initial_params nor"):
            optimize_parameters(qumat, z0_cost, {"theta": 0.0})

        params = {"theta": 0.1, "phi": 0.1}
        with pytest.raises(ValueError, match="learning_rate"):
            optimize_parameters(qumat, z0_cost, params, learning_rate=0)
        with pytest.raises(ValueError, match="max_iter"):
            optimize_parameters(qumat, z0_cost, params, max_iter=0)
        with pytest.raises(ValueError, match="unknown gradient"):
            optimize_parameters(qumat, z0_cost, params, gradient="autograd")
        with pytest.raises(ValueError, match="unsupported options for adam"):
            optimize_parameters(
                qumat, z0_cost, params, method="adam", options={"momentum": 0.9}
            )
        with pytest.raises(ValueError, match="non-finite"):
            optimize_parameters(qumat, lambda qm, p: float("nan"), params)
        # Unknown names fall through to scipy, which rejects them.
        with pytest.raises(ValueError, match="(?i)unknown solver"):
            optimize_parameters(qumat, z0_cost, params, method="not-a-method")

    def test_does_not_mutate_initial_params(self):
        qumat = make_qumat("cirq", 1)
        qumat.apply_ry_gate(0, "theta")
        initial = {"theta": 0.2}
        optimize_parameters(qumat, z0_cost, initial, max_iter=3)
        assert initial == {"theta": 0.2}
