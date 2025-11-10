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

from .utils import TESTING_BACKENDS, get_backend_config
from qumat import QuMat


def get_state_probability(results, target_state, num_qubits=1):
    """
    Calculate the probability of measuring a target state.

    Args:
        results: Dictionary of measurement results from execute_circuit()
        target_state: Target state as string (e.g., "0", "1", "101") or int
        num_qubits: Number of qubits in the circuit

    Returns:
        Probability of measuring the target state
    """
    if isinstance(results, list):
        results = results[0]

    total_shots = sum(results.values())
    if total_shots == 0:
        return 0.0

    # Convert target_state to both string and int formats for comparison
    if isinstance(target_state, str):
        target_str = target_state
        # Convert binary string to integer
        target_int = int(target_state, 2) if target_state else 0
    else:
        target_int = target_state
        # Convert integer to binary string
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
    """
    Calculate probabilities for |0⟩ and |1⟩ states in a superposition.

    Args:
        results: Dictionary of measurement results from execute_circuit()
        num_qubits: Number of qubits in the circuit

    Returns:
        Tuple of (prob_zero, prob_one)
    """
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
class TestPauliXGate:
    """Test class for Pauli X gate functionality."""

    @pytest.mark.parametrize(
        "initial_state, num_applications, expected_state",
        [
            ("0", 1, "1"),  # |0⟩ -> X -> |1⟩
            ("1", 1, "0"),  # |1⟩ -> X -> |0⟩
            ("0", 2, "0"),  # |0⟩ -> X -> X -> |0⟩
            ("1", 2, "1"),  # |1⟩ -> X -> X -> |1⟩
        ],
    )
    def test_pauli_x_state_transitions(
        self, backend_name, initial_state, num_applications, expected_state
    ):
        """Test Pauli X gate state transitions with parametrized test cases."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Prepare initial state: |0⟩ -> |initial_state⟩
        if initial_state == "1":
            qumat.apply_pauli_x_gate(0)  # |0⟩ -> |1⟩

        # Apply Pauli X gate specified number of times
        # This transforms |initial_state⟩ -> |expected_state⟩
        for _ in range(num_applications):
            qumat.apply_pauli_x_gate(0)

        # Execute circuit
        results = qumat.execute_circuit()

        # Calculate probability of expected state
        prob = get_state_probability(results, expected_state, num_qubits=1)

        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Initial state: |{initial_state}⟩, "
            f"Gate applications: {num_applications}, "
            f"Expected: |{expected_state}⟩, "
            f"Got probability: {prob:.4f}"
        )

    @pytest.mark.parametrize(
        "qubits_to_flip, num_qubits, expected_state",
        [
            ([0], 1, "1"),  # Single qubit: flip qubit 0 -> |1⟩
            ([0, 2], 3, "101"),  # Three qubits: flip qubits 0 and 2 -> |101⟩
            ([1], 3, "010"),  # Three qubits: flip qubit 1 -> |010⟩
            ([0, 1, 2], 3, "111"),  # Three qubits: flip all -> |111⟩
        ],
    )
    def test_pauli_x_on_multiple_qubits(
        self, backend_name, qubits_to_flip, num_qubits, expected_state
    ):
        """Test Pauli X gate on multiple qubits with parametrized test cases."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=num_qubits)

        # Apply Pauli X to specified qubits
        for qubit in qubits_to_flip:
            qumat.apply_pauli_x_gate(qubit)

        # Execute circuit
        results = qumat.execute_circuit()

        # Calculate probability of expected state
        prob = get_state_probability(results, expected_state, num_qubits=num_qubits)

        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Expected |{expected_state}⟩ state after flipping qubits "
            f"{qubits_to_flip}, got probability {prob:.4f}"
        )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestPauliYGate:
    """Test class for Pauli Y gate functionality."""

    @pytest.mark.parametrize(
        "initial_state, num_applications, expected_state",
        [
            ("0", 1, "1"),  # |0⟩ -> Y -> i|1⟩ (phase doesn't affect measurement)
            ("1", 1, "0"),  # |1⟩ -> Y -> -i|0⟩
            ("0", 2, "0"),  # |0⟩ -> Y -> Y -> |0⟩ (Y² = I)
            ("1", 2, "1"),  # |1⟩ -> Y -> Y -> |1⟩ (Y² = I)
        ],
    )
    def test_pauli_y_state_transitions(
        self, backend_name, initial_state, num_applications, expected_state
    ):
        """Test Pauli Y gate state transitions with parametrized test cases."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Prepare initial state: |0⟩ -> |initial_state⟩
        if initial_state == "1":
            qumat.apply_pauli_x_gate(0)  # |0⟩ -> |1⟩

        # Apply Pauli Y gate specified number of times
        # This transforms |initial_state⟩ -> |expected_state⟩
        for _ in range(num_applications):
            qumat.apply_pauli_y_gate(0)

        # Execute circuit
        results = qumat.execute_circuit()

        # Calculate probability of expected state
        prob = get_state_probability(results, expected_state, num_qubits=1)

        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Initial state: |{initial_state}⟩, "
            f"Gate applications: {num_applications}, "
            f"Expected: |{expected_state}⟩, "
            f"Got probability: {prob:.4f}"
        )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestHadamardGate:
    """Test class for Hadamard gate functionality."""

    @pytest.mark.parametrize(
        "initial_state, num_applications",
        [
            ("0", 1),  # |0⟩ -> H -> |+⟩ (superposition)
            ("1", 1),  # |1⟩ -> H -> |-⟩ (superposition)
            ("0", 2),  # |0⟩ -> H -> H -> |0⟩ (H² = I)
            ("1", 2),  # |1⟩ -> H -> H -> |1⟩ (H² = I)
        ],
    )
    def test_hadamard_state_transitions(
        self, backend_name, initial_state, num_applications
    ):
        """Test Hadamard gate state transitions with parametrized test cases."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Prepare initial state: |0⟩ -> |initial_state⟩
        if initial_state == "1":
            qumat.apply_pauli_x_gate(0)  # |0⟩ -> |1⟩

        # Apply Hadamard gate specified number of times
        # This transforms |initial_state⟩ -> |expected_state⟩
        for _ in range(num_applications):
            qumat.apply_hadamard_gate(0)

        # Execute circuit
        results = qumat.execute_circuit()

        if num_applications == 1:
            # Single application creates superposition
            prob_zero, prob_one = get_superposition_probabilities(results, num_qubits=1)
            assert 0.45 < prob_zero < 0.55, (
                f"Expected ~0.5 probability for |0⟩ after Hadamard on |{initial_state}⟩, "
                f"got {prob_zero}"
            )
            assert 0.45 < prob_one < 0.55, (
                f"Expected ~0.5 probability for |1⟩ after Hadamard on |{initial_state}⟩, "
                f"got {prob_one}"
            )
        else:
            # Double application returns to original state
            prob = get_state_probability(results, initial_state, num_qubits=1)
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Initial state: |{initial_state}⟩, "
                f"Gate applications: {num_applications}, "
                f"Expected: |{initial_state}⟩, "
                f"Got probability: {prob:.4f}"
            )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestNOTGate:
    """Test class for NOT gate functionality."""

    @pytest.mark.parametrize(
        "initial_state, num_applications, expected_state",
        [
            ("0", 1, "1"),  # |0⟩ -> NOT -> |1⟩ (equivalent to Pauli X)
            ("1", 1, "0"),  # |1⟩ -> NOT -> |0⟩
            ("0", 2, "0"),  # |0⟩ -> NOT -> NOT -> |0⟩
            ("1", 2, "1"),  # |1⟩ -> NOT -> NOT -> |1⟩
        ],
    )
    def test_not_gate_state_transitions(
        self, backend_name, initial_state, num_applications, expected_state
    ):
        """Test NOT gate state transitions with parametrized test cases."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Prepare initial state: |0⟩ -> |initial_state⟩
        if initial_state == "1":
            qumat.apply_not_gate(0)  # |0⟩ -> |1⟩

        # Apply NOT gate specified number of times
        # This transforms |initial_state⟩ -> |expected_state⟩
        for _ in range(num_applications):
            qumat.apply_not_gate(0)

        # Execute circuit
        results = qumat.execute_circuit()

        # Calculate probability of expected state
        prob = get_state_probability(results, expected_state, num_qubits=1)

        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Initial state: |{initial_state}⟩, "
            f"Gate applications: {num_applications}, "
            f"Expected: |{expected_state}⟩, "
            f"Got probability: {prob:.4f}"
        )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestUGate:
    """Test class for U gate (universal single-qubit gate) functionality."""

    @pytest.mark.parametrize(
        "theta, phi, lambd, expected_behavior",
        [
            (0, 0, 0, "identity"),  # U(0, 0, 0) should be identity
            (
                math.pi,
                0,
                math.pi,
                "pauli_x",
            ),  # U(π, 0, π) should be equivalent to Pauli X
            (
                math.pi / 2,
                0,
                math.pi,
                "hadamard",
            ),  # U(π/2, 0, π) should be equivalent to Hadamard
        ],
    )
    def test_u_gate_operations(
        self, backend_name, theta, phi, lambd, expected_behavior
    ):
        """Test U gate with different parameters using parametrized test cases."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply U gate with specified parameters
        qumat.apply_u_gate(0, theta=theta, phi=phi, lambd=lambd)

        # Execute circuit
        results = qumat.execute_circuit()

        if expected_behavior == "identity":
            # Should measure |0⟩ with high probability
            prob = get_state_probability(results, "0", num_qubits=1)
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Expected |0⟩ state after U({theta},{phi},{lambd}), got probability {prob:.4f}"
            )
        elif expected_behavior == "pauli_x":
            # Should measure |1⟩ with high probability
            prob = get_state_probability(results, "1", num_qubits=1)
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Expected |1⟩ state after U({theta},{phi},{lambd}), got probability {prob:.4f}"
            )
        elif expected_behavior == "hadamard":
            # Should have approximately equal probability for |0⟩ and |1⟩
            prob_zero, prob_one = get_superposition_probabilities(results, num_qubits=1)
            assert 0.45 < prob_zero < 0.55, (
                f"Expected ~0.5 probability for |0⟩ after U({theta},{phi},{lambd}), "
                f"got {prob_zero}"
            )
            assert 0.45 < prob_one < 0.55, (
                f"Expected ~0.5 probability for |1⟩ after U({theta},{phi},{lambd}), "
                f"got {prob_one}"
            )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestPauliZGate:
    """Test class for Pauli Z gate functionality."""

    @pytest.mark.parametrize(
        "initial_state, num_applications, expected_state",
        [
            ("0", 1, "0"),  # |0⟩ -> Z -> |0⟩ (Z leaves |0⟩ unchanged)
            ("1", 1, "1"),  # |1⟩ -> Z -> -|1⟩ (phase flip doesn't affect measurement)
            ("0", 2, "0"),  # |0⟩ -> Z -> Z -> |0⟩ (Z² = I)
            ("1", 2, "1"),  # |1⟩ -> Z -> Z -> |1⟩ (Z² = I)
        ],
    )
    def test_pauli_z_state_transitions(
        self, backend_name, initial_state, num_applications, expected_state
    ):
        """Test Pauli Z gate state transitions with parametrized test cases."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Prepare initial state: |0⟩ -> |initial_state⟩
        if initial_state == "1":
            qumat.apply_pauli_x_gate(0)  # |0⟩ -> |1⟩

        # Apply Pauli Z gate specified number of times
        # This transforms |initial_state⟩ -> |expected_state⟩
        for _ in range(num_applications):
            qumat.apply_pauli_z_gate(0)

        # Execute circuit
        results = qumat.execute_circuit()

        # Calculate probability of expected state
        prob = get_state_probability(results, expected_state, num_qubits=1)

        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Initial state: |{initial_state}⟩, "
            f"Gate applications: {num_applications}, "
            f"Expected: |{expected_state}⟩, "
            f"Got probability: {prob:.4f}"
        )

    def test_pauli_z_with_hadamard(self, backend_name):
        """Test Pauli Z gate with Hadamard to verify phase flip effect."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Create |+⟩ state with Hadamard
        qumat.apply_hadamard_gate(0)
        # Apply Pauli Z (should flip to |-⟩)
        qumat.apply_pauli_z_gate(0)
        # Apply Hadamard again (should convert |-⟩ to |1⟩)
        qumat.apply_hadamard_gate(0)

        # Execute circuit
        results = qumat.execute_circuit()

        # Calculate probability of |1⟩ state
        prob = get_state_probability(results, "1", num_qubits=1)

        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Expected |1⟩ state after H-Z-H sequence, got probability {prob:.4f}"
        )
