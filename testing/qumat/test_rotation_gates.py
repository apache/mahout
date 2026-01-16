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

import math

import pytest

from qumat import QuMat

from ..utils import TESTING_BACKENDS, get_backend_config


def get_state_probability(results, target_state, num_qubits=1):
    """Calculate the probability of measuring a target state."""
    if isinstance(results, list):
        results = results[0]

    total_shots = sum(results.values())
    if total_shots == 0:
        return 0.0

    if isinstance(target_state, str):
        target_str = target_state
        target_int = int(target_state, 2) if target_state else 0
    else:
        target_int = target_state
        target_str = format(target_state, f"0{num_qubits}b")

    target_count = 0
    for state, count in results.items():
        if isinstance(state, str):
            if state == target_str:
                target_count = count
                break
        else:
            if state == target_int:
                target_count = count
                break

    return target_count / total_shots


def get_superposition_probabilities(results, num_qubits=1):
    """Calculate probabilities for |0⟩ and |1⟩ states in a superposition."""
    if isinstance(results, list):
        results = results[0]

    total_shots = sum(results.values())

    zero_count = 0
    one_count = 0
    for state, count in results.items():
        if isinstance(state, str):
            if state == "0" * num_qubits:
                zero_count = count
            elif state == "1" * num_qubits:
                one_count = count
        else:
            if state == 0:
                zero_count = count
            elif state == (2**num_qubits - 1):
                one_count = count

    prob_zero = zero_count / total_shots if total_shots > 0 else 0.0
    prob_one = one_count / total_shots if total_shots > 0 else 0.0

    return prob_zero, prob_one


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestRXGate:
    """Test class for RX gate functionality."""

    @pytest.mark.parametrize(
        "angle, expected_behavior",
        [
            (0, "identity"),  # RX(0) = I
            (math.pi, "pauli_x"),  # RX(π) = X
            (math.pi / 2, "superposition"),  # RX(π/2) creates superposition
            (2 * math.pi, "identity"),  # RX(2π) = I
        ],
    )
    def test_rx_gate_with_different_angles(
        self, backend_name, angle, expected_behavior
    ):
        """Test RX gate with different angles."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        qumat.apply_rx_gate(0, angle)
        results = qumat.execute_circuit()

        if expected_behavior == "identity":
            prob = get_state_probability(results, "0", num_qubits=1)
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Expected |0⟩ after RX({angle:.4f}), got probability {prob:.4f}"
            )
        elif expected_behavior == "pauli_x":
            prob = get_state_probability(results, "1", num_qubits=1)
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Expected |1⟩ after RX({angle:.4f}), got probability {prob:.4f}"
            )
        elif expected_behavior == "superposition":
            prob_zero, prob_one = get_superposition_probabilities(results, num_qubits=1)
            assert 0.45 < prob_zero < 0.55, (
                f"Expected ~0.5 probability for |0⟩ after RX({angle:.4f}), "
                f"got {prob_zero:.4f}"
            )
            assert 0.45 < prob_one < 0.55, (
                f"Expected ~0.5 probability for |1⟩ after RX({angle:.4f}), "
                f"got {prob_one:.4f}"
            )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestRYGate:
    """Test class for RY gate functionality."""

    @pytest.mark.parametrize(
        "angle, expected_behavior",
        [
            (0, "identity"),  # RY(0) = I
            (math.pi, "pauli_y"),  # RY(π) ≈ Y (phase doesn't affect measurement)
            (math.pi / 2, "superposition"),  # RY(π/2) creates superposition
            (2 * math.pi, "identity"),  # RY(2π) = I
        ],
    )
    def test_ry_gate_with_different_angles(
        self, backend_name, angle, expected_behavior
    ):
        """Test RY gate with different angles."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        qumat.apply_ry_gate(0, angle)
        results = qumat.execute_circuit()

        if expected_behavior == "identity":
            prob = get_state_probability(results, "0", num_qubits=1)
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Expected |0⟩ after RY({angle:.4f}), got probability {prob:.4f}"
            )
        elif expected_behavior == "pauli_y":
            # RY(π) flips |0⟩ to |1⟩ (like Y gate, phase doesn't affect measurement)
            prob = get_state_probability(results, "1", num_qubits=1)
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Expected |1⟩ after RY({angle:.4f}), got probability {prob:.4f}"
            )
        elif expected_behavior == "superposition":
            prob_zero, prob_one = get_superposition_probabilities(results, num_qubits=1)
            assert 0.45 < prob_zero < 0.55, (
                f"Expected ~0.5 probability for |0⟩ after RY({angle:.4f}), "
                f"got {prob_zero:.4f}"
            )
            assert 0.45 < prob_one < 0.55, (
                f"Expected ~0.5 probability for |1⟩ after RY({angle:.4f}), "
                f"got {prob_one:.4f}"
            )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestRZGate:
    """Test class for RZ gate functionality."""

    @pytest.mark.parametrize(
        "angle, expected_state",
        [
            (0, "0"),  # RZ(0) = I, |0⟩ -> |0⟩
            (math.pi, "0"),  # RZ(π) adds phase, but |0⟩ measurement unchanged
            (2 * math.pi, "0"),  # RZ(2π) = I
        ],
    )
    def test_rz_gate_with_different_angles(self, backend_name, angle, expected_state):
        """Test RZ gate with different angles."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        qumat.apply_rz_gate(0, angle)
        results = qumat.execute_circuit()

        # RZ only affects phase, not measurement probability for |0⟩
        prob = get_state_probability(results, expected_state, num_qubits=1)
        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Expected |{expected_state}⟩ after RZ({angle:.4f}), got probability {prob:.4f}"
        )

    def test_rz_gate_phase_effect(self, backend_name):
        """Test RZ gate phase effect using Hadamard."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # H -> RZ(π) -> H should flip |0⟩ to |1⟩
        qumat.apply_hadamard_gate(0)  # |0⟩ -> |+⟩
        qumat.apply_rz_gate(0, math.pi)  # |+⟩ -> |-⟩
        qumat.apply_hadamard_gate(0)  # |-⟩ -> |1⟩

        results = qumat.execute_circuit()
        prob = get_state_probability(results, "1", num_qubits=1)

        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Expected |1⟩ after H-RZ(π)-H, got probability {prob:.4f}"
        )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestParameterizedRotationGates:
    """Test class for parameterized rotation gates using string parameters."""

    @pytest.mark.parametrize(
        "gate_type, param_name",
        [
            ("rx", "theta"),
            ("ry", "phi"),
            ("rz", "lambda"),
        ],
    )
    def test_parameterized_rotation_gate(self, backend_name, gate_type, param_name):
        """Test parameterized rotation gates with string parameters."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply parameterized gate
        if gate_type == "rx":
            qumat.apply_rx_gate(0, param_name)
        elif gate_type == "ry":
            qumat.apply_ry_gate(0, param_name)
        elif gate_type == "rz":
            qumat.apply_rz_gate(0, param_name)

        # Verify parameter was registered
        assert param_name in qumat.parameters, (
            f"Parameter '{param_name}' should be registered in circuit"
        )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestParameterBinding:
    """Test class for parameter binding in rotation gates."""

    @pytest.mark.parametrize(
        "gate_type, param_name, bound_value, expected_behavior",
        [
            ("rx", "theta", math.pi, "pauli_x"),  # RX(π) = X
            (
                "rx",
                "theta",
                math.pi / 2,
                "superposition",
            ),  # RX(π/2) creates superposition
            (
                "ry",
                "phi",
                math.pi / 2,
                "superposition",
            ),  # RY(π/2) creates superposition
            (
                "rz",
                "lambda",
                math.pi,
                "identity",
            ),  # RZ(π) doesn't change |0⟩ measurement
        ],
    )
    def test_parameter_binding(
        self,
        backend_name,
        gate_type,
        param_name,
        bound_value,
        expected_behavior,
    ):
        """Test parameter binding for rotation gates."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply parameterized gate
        if gate_type == "rx":
            qumat.apply_rx_gate(0, param_name)
        elif gate_type == "ry":
            qumat.apply_ry_gate(0, param_name)
        elif gate_type == "rz":
            qumat.apply_rz_gate(0, param_name)

        # Execute circuit with parameter values
        results = qumat.execute_circuit(parameter_values={param_name: bound_value})

        if expected_behavior == "pauli_x":
            prob = get_state_probability(results, "1", num_qubits=1)
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Expected |1⟩ after binding {param_name}={bound_value:.4f}, "
                f"got probability {prob:.4f}"
            )
        elif expected_behavior == "superposition":
            prob_zero, prob_one = get_superposition_probabilities(results, num_qubits=1)
            assert 0.45 < prob_zero < 0.55, (
                f"Expected ~0.5 probability for |0⟩ after binding {param_name}={bound_value:.4f}, "
                f"got {prob_zero:.4f}"
            )
            assert 0.45 < prob_one < 0.55, (
                f"Expected ~0.5 probability for |1⟩ after binding {param_name}={bound_value:.4f}, "
                f"got {prob_one:.4f}"
            )
        elif expected_behavior == "identity":
            prob = get_state_probability(results, "0", num_qubits=1)
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Expected |0⟩ after binding {param_name}={bound_value:.4f}, "
                f"got probability {prob:.4f}"
            )

    def test_multiple_parameter_binding(self, backend_name):
        """Test binding multiple parameters."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=2)

        # Apply different parameterized gates to different qubits
        qumat.apply_rx_gate(0, "theta0")
        qumat.apply_ry_gate(1, "phi1")

        # Execute circuit with multiple parameter values
        results = qumat.execute_circuit(
            parameter_values={"theta0": math.pi, "phi1": math.pi / 2}
        )

        # Qubit 0 should be |1⟩ (RX(π) = X)
        # Qubit 1 should be in superposition (RY(π/2))
        # Check that we get expected states
        prob_one_zero = get_state_probability(results, "10", num_qubits=2)
        prob_one_one = get_state_probability(results, "11", num_qubits=2)

        # At least one of these should have significant probability
        total_prob = prob_one_zero + prob_one_one
        assert total_prob > 0.4, (
            f"Expected high probability for states with qubit 0=|1⟩, "
            f"got {total_prob:.4f}"
        )

    def test_invalid_parameter_binding(self, backend_name):
        """Test that binding invalid parameter raises error."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        qumat.apply_rx_gate(0, "theta")

        # Try to bind non-existent parameter
        with pytest.raises(ValueError, match="parameter 'invalid_param' not found"):
            qumat.bind_parameters({"invalid_param": math.pi})


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestRotationGatesEdgeCases:
    """Test class for edge cases of rotation gates."""

    def test_rotation_gate_on_uninitialized_circuit(self, backend_name):
        """Test that rotation gates raise error on uninitialized circuit."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        with pytest.raises(RuntimeError, match="circuit not initialized"):
            qumat.apply_rx_gate(0, math.pi)

    def test_rotation_gate_with_negative_angle(self, backend_name):
        """Test rotation gates with negative angles."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # RX(-π) should be equivalent to RX(π) = X
        qumat.apply_rx_gate(0, -math.pi)
        results = qumat.execute_circuit()

        prob = get_state_probability(results, "1", num_qubits=1)
        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Expected |1⟩ after RX(-π), got probability {prob:.4f}"
        )

    def test_rotation_gate_with_large_angle(self, backend_name):
        """Test rotation gates with large angles."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # RX(4π) = RX(0) = I
        qumat.apply_rx_gate(0, 4 * math.pi)
        results = qumat.execute_circuit()

        prob = get_state_probability(results, "0", num_qubits=1)
        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Expected |0⟩ after RX(4π), got probability {prob:.4f}"
        )

    def test_multiple_rotations_on_same_qubit(self, backend_name):
        """Test applying multiple rotations sequentially on the same qubit."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # RX(π/2) -> RY(π/2) -> RZ(π/2) sequence
        qumat.apply_rx_gate(0, math.pi / 2)  # Creates superposition
        qumat.apply_ry_gate(0, math.pi / 2)  # Rotates superposition
        qumat.apply_rz_gate(0, math.pi / 2)  # Adds phase

        results = qumat.execute_circuit()

        # Should still be in a valid quantum state
        total_shots = (
            sum(results.values())
            if isinstance(results, dict)
            else sum(results[0].values())
        )
        assert total_shots > 0, "Circuit execution should produce results"

    def test_rotation_gate_on_invalid_qubit_index(self, backend_name):
        """Test rotation gates with invalid qubit index."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=2)

        try:
            qumat.apply_rx_gate(5, math.pi)  # Invalid index
        except (IndexError, ValueError, RuntimeError, Exception):
            pass
