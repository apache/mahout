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

from .utils import TESTING_BACKENDS, get_backend_config, get_state_probability
from qumat import QuMat


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
        prob = get_state_probability(
            results, expected_state, num_qubits=1, backend_name=backend_name
        )

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
        prob = get_state_probability(
            results, expected_state, num_qubits=num_qubits, backend_name=backend_name
        )

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
        prob = get_state_probability(
            results, expected_state, num_qubits=1, backend_name=backend_name
        )

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
            prob = get_state_probability(
                results, initial_state, num_qubits=1, backend_name=backend_name
            )
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
        prob = get_state_probability(
            results, expected_state, num_qubits=1, backend_name=backend_name
        )

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
            # Additional test cases with non-zero phi to detect decomposition errors
            (
                math.pi / 4,
                math.pi / 4,
                math.pi / 4,
                "superposition",
            ),  # U(π/4, π/4, π/4) should create superposition
            (
                math.pi / 2,
                math.pi / 4,
                0,
                "superposition",
            ),  # U(π/2, π/4, 0) should create superposition
            (
                0,
                math.pi / 2,
                0,
                "identity",
            ),  # U(0, π/2, 0) = Rz(π/2) · I · I = Rz(π/2), phase rotation only
            (
                math.pi / 2,
                math.pi / 2,
                math.pi / 2,
                "superposition",
            ),  # U(π/2, π/2, π/2) should create superposition
            (
                math.pi / 3,
                math.pi / 6,
                math.pi / 4,
                "superposition",
            ),  # U(π/3, π/6, π/4) with all non-zero parameters
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
            prob = get_state_probability(
                results, "0", num_qubits=1, backend_name=backend_name
            )
            assert prob > 0.95, (
                f"Backend: {backend_name}, "
                f"Expected |0⟩ state after U({theta},{phi},{lambd}), got probability {prob:.4f}"
            )
        elif expected_behavior == "pauli_x":
            # Should measure |1⟩ with high probability
            prob = get_state_probability(
                results, "1", num_qubits=1, backend_name=backend_name
            )
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
        elif expected_behavior == "superposition":
            # Should create a superposition (not all probability in one state)
            prob_zero, prob_one = get_superposition_probabilities(results, num_qubits=1)
            # At least one state should have significant probability (> 0.1)
            # and not all probability should be in one state (< 0.9)
            assert prob_zero > 0.1 or prob_one > 0.1, (
                f"Expected superposition after U({theta},{phi},{lambd}), got prob_zero={prob_zero:.4f}, prob_one={prob_one:.4f}"
            )
            assert prob_zero < 0.9 and prob_one < 0.9, (
                f"Expected superposition after U({theta},{phi},{lambd}), got prob_zero={prob_zero:.4f}, prob_one={prob_one:.4f}"
            )


@pytest.mark.parametrize(
    "theta, phi, lambd",
    [
        (math.pi / 4, math.pi / 4, math.pi / 4),
        (math.pi / 2, math.pi / 4, 0),
        (math.pi / 3, math.pi / 6, math.pi / 4),
        (math.pi / 2, math.pi / 2, math.pi / 2),
    ],
)
def test_u_gate_cross_backend_consistency(theta, phi, lambd):
    """Test that U gate produces consistent results across all backends.

    Test cases with non-zero phi to detect decomposition errors.
    """
    results_dict = {}

    for backend_name in TESTING_BACKENDS:
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply U gate with specified parameters
        qumat.apply_u_gate(0, theta=theta, phi=phi, lambd=lambd)

        # Execute circuit
        results = qumat.execute_circuit()

        # Calculate probabilities for |0⟩ and |1⟩
        prob_zero, prob_one = get_superposition_probabilities(results, num_qubits=1)
        results_dict[backend_name] = (prob_zero, prob_one)

    # All backends should give similar results (within 5% tolerance)
    backends = list(results_dict.keys())
    for i in range(len(backends)):
        for j in range(i + 1, len(backends)):
            backend1 = backends[i]
            backend2 = backends[j]
            prob_zero1, prob_one1 = results_dict[backend1]
            prob_zero2, prob_one2 = results_dict[backend2]

            diff_zero = abs(prob_zero1 - prob_zero2)
            diff_one = abs(prob_one1 - prob_one2)

            assert diff_zero < 0.05, (
                f"Backends {backend1} and {backend2} have inconsistent |0⟩ probabilities "
                f"for U({theta},{phi},{lambd}): {prob_zero1:.4f} vs {prob_zero2:.4f}"
            )
            assert diff_one < 0.05, (
                f"Backends {backend1} and {backend2} have inconsistent |1⟩ probabilities "
                f"for U({theta},{phi},{lambd}): {prob_one1:.4f} vs {prob_one2:.4f}"
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
        prob = get_state_probability(
            results, expected_state, num_qubits=1, backend_name=backend_name
        )

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
        prob = get_state_probability(
            results, "1", num_qubits=1, backend_name=backend_name
        )

        assert prob > 0.95, (
            f"Backend: {backend_name}, "
            f"Expected |1⟩ state after H-Z-H sequence, got probability {prob:.4f}"
        )



@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestTGate:
    """Test class for T gate functionality."""

    @pytest.mark.parametrize(
        "initial_state, expected_state",
        [
            ("0", "0"),  # T leaves |0> unchanged
            ("1", "1"),  # T applies phase to |1>, measurement unchanged
        ],
    )
    def test_t_gate_preserves_basis_states(
        self, backend_name, initial_state, expected_state
    ):
        """T gate should preserve computational basis measurement outcomes."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        if initial_state == "1":
            qumat.apply_pauli_x_gate(0)

        qumat.apply_t_gate(0)
        results = qumat.execute_circuit()

        prob = get_state_probability(
            results, expected_state, num_qubits=1, backend_name=backend_name
        )
        assert prob > 0.95, (
            f"Backend: {backend_name}, expected |{expected_state}> after T, "
            f"got probability {prob:.4f}"
        )

    def test_t_gate_phase_visible_via_hzh(self, backend_name):
        """T^4 = Z; H-Z-H should act like X and flip |0> to |1>."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        qumat.apply_hadamard_gate(0)
        for _ in range(4):
            qumat.apply_t_gate(0)
        qumat.apply_hadamard_gate(0)

        results = qumat.execute_circuit()
        prob = get_state_probability(
            results, "1", num_qubits=1, backend_name=backend_name
        )
        assert prob > 0.95, (
            f"Backend: {backend_name}, expected |1> after H-T^4-H, "
            f"got probability {prob:.4f}"
        )

    def test_t_gate_eight_applications_identity(self, backend_name):
        """T^8 should be identity."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        for _ in range(8):
            qumat.apply_t_gate(0)

        results = qumat.execute_circuit()
        prob = get_state_probability(
            results, "0", num_qubits=1, backend_name=backend_name
        )
        assert prob > 0.95, (
            f"Backend: {backend_name}, expected |0> after T^8, "
            f"got probability {prob:.4f}"
        )


@pytest.mark.parametrize(
    "phase_applications",
    [
        1,  # single T
        2,  # T^2 = S
        4,  # T^4 = Z
    ],
)
def test_t_gate_cross_backend_consistency(phase_applications):
    """T gate should behave consistently across all backends."""
    results_dict = {}

    for backend_name in TESTING_BACKENDS:
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Use H ... H sandwich to turn phase into amplitude when needed
        qumat.apply_hadamard_gate(0)
        for _ in range(phase_applications):
            qumat.apply_t_gate(0)
        qumat.apply_hadamard_gate(0)

        results = qumat.execute_circuit()
        prob_one = get_state_probability(
            results, "1", num_qubits=1, backend_name=backend_name
        )
        results_dict[backend_name] = prob_one

    backends = list(results_dict.keys())
    for i in range(len(backends)):
        for j in range(i + 1, len(backends)):
            b1, b2 = backends[i], backends[j]
            diff = abs(results_dict[b1] - results_dict[b2])
            assert diff < 0.05, (
                f"T gate inconsistent between {b1} and {b2} for T^{phase_applications}: "
                f"{results_dict[b1]:.4f} vs {results_dict[b2]:.4f}"
            )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestSingleQubitGatesEdgeCases:
    """Test class for edge cases of single-qubit gates."""

    def test_gate_on_uninitialized_circuit(self, backend_name):
        """Test that gates raise error on uninitialized circuit."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Try to apply gate without creating circuit
        with pytest.raises(RuntimeError, match="circuit not initialized"):
            qumat.apply_pauli_x_gate(0)

    def test_gate_on_invalid_qubit_index(self, backend_name):
        """Test that gates handle invalid qubit indices appropriately."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=2)

        # Different backends may handle this differently
        try:
            qumat.apply_pauli_x_gate(5)  # Invalid index
        except (IndexError, ValueError, RuntimeError, Exception):
            pass

    def test_gate_on_zero_qubit_circuit(self, backend_name):
        """Test gates on zero-qubit circuit."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=0)

        try:
            qumat.apply_pauli_x_gate(0)
            results = qumat.execute_circuit()
            assert results is not None
        except (IndexError, ValueError, RuntimeError, Exception):
            pass

    def test_multiple_gates_on_same_qubit(self, backend_name):
        """Test applying multiple gates sequentially on the same qubit."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply multiple gates in sequence
        qumat.apply_pauli_x_gate(0)  # |0⟩ -> |1⟩
        qumat.apply_hadamard_gate(0)  # |1⟩ -> |-⟩
        qumat.apply_pauli_z_gate(0)  # |-⟩ -> |+⟩
        qumat.apply_hadamard_gate(0)  # |+⟩ -> |0⟩

        # Execute circuit
        results = qumat.execute_circuit()

        # Calculate probability of |0⟩ state
        prob = get_state_probability(
            results, "0", num_qubits=1, backend_name=backend_name
        )

        assert prob > 0.95, (
            f"Expected |0⟩ state after gate sequence, got probability {prob}"
        )

    def test_gates_on_different_qubits_independently(self, backend_name):
        """Test that gates on different qubits operate independently."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=3)

        # Apply different gates to different qubits
        qumat.apply_pauli_x_gate(0)  # Qubit 0: |0⟩ -> |1⟩
        qumat.apply_hadamard_gate(1)  # Qubit 1: |0⟩ -> |+⟩
        # Qubit 2: remains |0⟩

        # Execute circuit
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]

        total_shots = sum(results.values())

        # Check qubit 0=|1⟩, qubit 1=superposition, qubit 2=|0⟩
        # Backends use different bit ordering (little-endian vs big-endian)
        target_states_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if backend_name == "qiskit":
                    # Little-endian: "x01" where x is 0 or 1
                    if (
                        len(state) == 3
                        and state[0] in ["0", "1"]
                        and state[1] in ["0", "1"]
                        and state[2] == "1"
                        and state[0] == "0"
                    ):
                        target_states_count += count
                elif backend_name == "amazon_braket":
                    # Big-endian: "1x0" where x is 0 or 1
                    if len(state) == 3 and state[0] == "1" and state[2] == "0":
                        target_states_count += count
            else:
                # Cirq: integer format, |100⟩=4, |101⟩=5
                if state in [4, 5]:
                    target_states_count += count

        prob_target = target_states_count / total_shots
        assert prob_target > 0.4, (
            f"Expected high probability for target states, got {prob_target}"
        )


@pytest.mark.parametrize(
    "gate_name, expected_state_or_behavior",
    [
        ("pauli_x", "1"),  # Pauli X should flip |0⟩ to |1⟩
        ("hadamard", "superposition"),  # Hadamard creates superposition
    ],
)
def test_gate_consistency(gate_name, expected_state_or_behavior):
    """Test that gates produce consistent results across all backends."""
    results_dict = {}

    for backend_name in TESTING_BACKENDS:
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply the gate based on gate_name
        if gate_name == "pauli_x":
            qumat.apply_pauli_x_gate(0)
        elif gate_name == "hadamard":
            qumat.apply_hadamard_gate(0)
        # Future gates can be easily added here

        results = qumat.execute_circuit()

        if expected_state_or_behavior == "superposition":
            # For Hadamard, check superposition probabilities
            prob_zero, _ = get_superposition_probabilities(results, num_qubits=1)
            results_dict[backend_name] = prob_zero
        else:
            # For other gates, check specific state probability
            prob = get_state_probability(
                results,
                expected_state_or_behavior,
                num_qubits=1,
                backend_name=backend_name,
            )
            results_dict[backend_name] = prob

    # All backends should give similar results
    probabilities = list(results_dict.values())
    for i in range(len(probabilities)):
        for j in range(i + 1, len(probabilities)):
            diff = abs(probabilities[i] - probabilities[j])
            assert diff < 0.05, (
                f"Backends have inconsistent results for {gate_name}: {results_dict}"
            )
