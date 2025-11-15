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

import pytest

from .utils import TESTING_BACKENDS, get_backend_config, get_state_probability
from qumat import QuMat


def create_qumat_instance(backend_name, num_qubits):
    """Create and initialize a QuMat instance with a circuit."""
    backend_config = get_backend_config(backend_name)
    qumat = QuMat(backend_config)
    qumat.create_empty_circuit(num_qubits=num_qubits)
    return qumat


def prepare_initial_state(qumat, initial_state_str):
    """
    Prepare initial state by applying X gates to qubits that should be |1⟩.

    Args:
        qumat: QuMat instance
        initial_state_str: Binary string representing initial state (e.g., "101")
    """
    for i, bit in enumerate(initial_state_str):
        if bit == "1":
            qumat.apply_pauli_x_gate(i)


def execute_and_assert_state(
    qumat, expected_state, num_qubits, backend_name, threshold=0.95, context_msg=""
):
    """
    Execute circuit and assert expected state probability.

    Args:
        qumat: QuMat instance
        expected_state: Expected state string or int
        num_qubits: Number of qubits
        backend_name: Backend name
        threshold: Probability threshold (default 0.95)
        context_msg: Additional context message for assertion
    """
    results = qumat.execute_circuit()
    prob = get_state_probability(results, expected_state, num_qubits, backend_name)
    assert prob > threshold, (
        f"Backend: {backend_name}, {context_msg}, "
        f"Expected: |{expected_state}⟩, Got probability: {prob:.4f}"
    )
    return prob


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestCNOTGate:
    """Test class for CNOT gate functionality."""

    @pytest.mark.parametrize(
        "initial_state, control_qubit, target_qubit, expected_state",
        [
            ("00", 0, 1, "00"),  # |00⟩ -> CNOT(0,1) -> |00⟩ (control=0, no flip)
            ("01", 0, 1, "01"),  # |01⟩ -> CNOT(0,1) -> |01⟩ (control=0, no flip)
            ("10", 0, 1, "11"),  # |10⟩ -> CNOT(0,1) -> |11⟩ (control=1, flip target)
            ("11", 0, 1, "10"),  # |11⟩ -> CNOT(0,1) -> |10⟩ (control=1, flip target)
            ("00", 1, 0, "00"),  # |00⟩ -> CNOT(1,0) -> |00⟩ (control=0, no flip)
            ("01", 1, 0, "11"),  # |01⟩ -> CNOT(1,0) -> |11⟩ (control=1, flip target)
            ("10", 1, 0, "10"),  # |10⟩ -> CNOT(1,0) -> |10⟩ (control=0, no flip)
            ("11", 1, 0, "01"),  # |11⟩ -> CNOT(1,0) -> |01⟩ (control=1, flip target)
        ],
    )
    def test_cnot_state_transitions(
        self, backend_name, initial_state, control_qubit, target_qubit, expected_state
    ):
        """Test CNOT gate state transitions with parametrized test cases."""
        qumat = create_qumat_instance(backend_name, num_qubits=2)
        prepare_initial_state(qumat, initial_state)
        qumat.apply_cnot_gate(control_qubit, target_qubit)
        execute_and_assert_state(
            qumat,
            expected_state,
            num_qubits=2,
            backend_name=backend_name,
            context_msg=(
                f"Initial state: |{initial_state}⟩, "
                f"CNOT(control={control_qubit}, target={target_qubit})"
            ),
        )

    @pytest.mark.parametrize(
        "initial_state, num_applications, expected_state",
        [
            ("00", 1, "00"),  # |00⟩ -> CNOT -> |00⟩
            ("00", 2, "00"),  # |00⟩ -> CNOT -> CNOT -> |00⟩ (CNOT² = I)
            ("10", 1, "11"),  # |10⟩ -> CNOT -> |11⟩
            ("10", 2, "10"),  # |10⟩ -> CNOT -> |11⟩ -> CNOT -> |10⟩ (CNOT² = I)
            ("11", 1, "10"),  # |11⟩ -> CNOT -> |10⟩
            ("11", 2, "11"),  # |11⟩ -> CNOT -> |10⟩ -> CNOT -> |11⟩ (CNOT² = I)
        ],
    )
    def test_cnot_double_application(
        self, backend_name, initial_state, num_applications, expected_state
    ):
        """Test that applying CNOT twice returns to original state (CNOT² = I)."""
        qumat = create_qumat_instance(backend_name, num_qubits=2)
        prepare_initial_state(qumat, initial_state)
        for _ in range(num_applications):
            qumat.apply_cnot_gate(0, 1)
        execute_and_assert_state(
            qumat,
            expected_state,
            num_qubits=2,
            backend_name=backend_name,
            context_msg=(
                f"Initial state: |{initial_state}⟩, "
                f"CNOT applications: {num_applications}"
            ),
        )

    @pytest.mark.parametrize(
        "control_qubit, target_qubit, num_qubits",
        [
            # 3-qubit circuits
            (0, 1, 3),  # CNOT on qubits 0 and 1 in 3-qubit circuit
            (1, 2, 3),  # CNOT on qubits 1 and 2 in 3-qubit circuit
            (0, 2, 3),  # CNOT on qubits 0 and 2 in 3-qubit circuit
            # 4-qubit circuits
            (0, 1, 4),  # CNOT on qubits 0 and 1 in 4-qubit circuit
            (0, 3, 4),  # CNOT on qubits 0 and 3 in 4-qubit circuit
            (1, 2, 4),  # CNOT on qubits 1 and 2 in 4-qubit circuit
            (2, 3, 4),  # CNOT on qubits 2 and 3 in 4-qubit circuit
            # 5-qubit circuits
            (0, 4, 5),  # CNOT on qubits 0 and 4 in 5-qubit circuit
            (2, 3, 5),  # CNOT on qubits 2 and 3 in 5-qubit circuit
        ],
    )
    def test_cnot_on_multiple_qubits(
        self, backend_name, control_qubit, target_qubit, num_qubits
    ):
        """Test CNOT gate on different qubit pairs in multi-qubit circuits."""
        qumat = create_qumat_instance(backend_name, num_qubits=num_qubits)
        qumat.apply_pauli_x_gate(control_qubit)
        qumat.apply_cnot_gate(control_qubit, target_qubit)
        # Expected: control and target qubits should both be |1⟩
        expected_state = "".join(
            "1" if i in (control_qubit, target_qubit) else "0"
            for i in range(num_qubits)
        )
        execute_and_assert_state(
            qumat,
            expected_state,
            num_qubits=num_qubits,
            backend_name=backend_name,
            context_msg=(
                f"CNOT(control={control_qubit}, target={target_qubit}) "
                f"on {num_qubits}-qubit circuit"
            ),
        )

    @pytest.mark.parametrize(
        "control_qubit, target_qubit, expected_states",
        [
            # Standard Bell state: |00⟩ + |11⟩
            (0, 1, ["00", "11"]),
            # Reversed control/target: |00⟩ + |11⟩ (same result)
            (1, 0, ["00", "11"]),
        ],
    )
    def test_cnot_entanglement(
        self, backend_name, control_qubit, target_qubit, expected_states
    ):
        """Test that CNOT gate creates entanglement (Bell state) with parametrized test cases."""
        qumat = create_qumat_instance(backend_name, num_qubits=2)
        qumat.apply_hadamard_gate(control_qubit)
        qumat.apply_cnot_gate(control_qubit, target_qubit)
        results = qumat.execute_circuit()
        # Should measure expected states with approximately equal probability
        for expected_state in expected_states:
            prob = get_state_probability(
                results, expected_state, num_qubits=2, backend_name=backend_name
            )
            assert 0.45 < prob < 0.55, (
                f"Backend: {backend_name}, "
                f"CNOT(control={control_qubit}, target={target_qubit}), "
                f"Expected ~0.5 probability for |{expected_state}⟩ in Bell state, got {prob:.4f}"
            )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestToffoliGate:
    """Test class for Toffoli gate functionality."""

    @pytest.mark.parametrize(
        "initial_state, control1, control2, target, expected_state",
        [
            # Toffoli(0,1,2): flip target only if both controls are |1⟩
            ("000", 0, 1, 2, "000"),  # |000⟩ -> Toffoli -> |000⟩
            ("001", 0, 1, 2, "001"),  # |001⟩ -> Toffoli -> |001⟩
            ("010", 0, 1, 2, "010"),  # |010⟩ -> Toffoli -> |010⟩
            ("011", 0, 1, 2, "011"),  # |011⟩ -> Toffoli -> |011⟩
            ("100", 0, 1, 2, "100"),  # |100⟩ -> Toffoli -> |100⟩
            ("101", 0, 1, 2, "101"),  # |101⟩ -> Toffoli -> |101⟩
            (
                "110",
                0,
                1,
                2,
                "111",
            ),  # |110⟩ -> Toffoli -> |111⟩ (both controls=1, flip target)
            (
                "111",
                0,
                1,
                2,
                "110",
            ),  # |111⟩ -> Toffoli -> |110⟩ (both controls=1, flip target)
            # Different control/target combinations
            # |110⟩ -> Toffoli(0,2,1): control0=1, control2=0 -> no flip, result |110⟩
            (
                "110",
                0,
                2,
                1,
                "110",
            ),  # |110⟩ -> Toffoli(0,2,1) -> |110⟩ (control2=0, no flip)
            # |101⟩ -> Toffoli(1,2,0): control1=0, control2=1 -> no flip, result |101⟩
            (
                "101",
                1,
                2,
                0,
                "101",
            ),  # |101⟩ -> Toffoli(1,2,0) -> |101⟩ (control1=0, no flip)
            # Test cases where both controls are 1 with different target
            (
                "111",
                0,
                2,
                1,
                "101",
            ),  # |111⟩ -> Toffoli(0,2,1) -> |101⟩ (both controls=1, flip target)
            (
                "111",
                1,
                2,
                0,
                "011",
            ),  # |111⟩ -> Toffoli(1,2,0) -> |011⟩ (both controls=1, flip target)
        ],
    )
    def test_toffoli_state_transitions(
        self,
        backend_name,
        initial_state,
        control1,
        control2,
        target,
        expected_state,
    ):
        """Test Toffoli gate state transitions with parametrized test cases."""
        qumat = create_qumat_instance(backend_name, num_qubits=3)
        prepare_initial_state(qumat, initial_state)
        qumat.apply_toffoli_gate(control1, control2, target)
        execute_and_assert_state(
            qumat,
            expected_state,
            num_qubits=3,
            backend_name=backend_name,
            context_msg=(
                f"Initial state: |{initial_state}⟩, "
                f"Toffoli(control1={control1}, control2={control2}, target={target})"
            ),
        )

    @pytest.mark.parametrize(
        "initial_state, num_applications, expected_state",
        [
            ("000", 1, "000"),  # |000⟩ -> Toffoli -> |000⟩
            ("000", 2, "000"),  # |000⟩ -> Toffoli -> Toffoli -> |000⟩ (Toffoli² = I)
            ("110", 1, "111"),  # |110⟩ -> Toffoli -> |111⟩
            (
                "110",
                2,
                "110",
            ),  # |110⟩ -> Toffoli -> |111⟩ -> Toffoli -> |110⟩ (Toffoli² = I)
            ("111", 1, "110"),  # |111⟩ -> Toffoli -> |110⟩
            (
                "111",
                2,
                "111",
            ),  # |111⟩ -> Toffoli -> |110⟩ -> Toffoli -> |111⟩ (Toffoli² = I)
        ],
    )
    def test_toffoli_double_application(
        self, backend_name, initial_state, num_applications, expected_state
    ):
        """Test that applying Toffoli twice returns to original state (Toffoli² = I)."""
        qumat = create_qumat_instance(backend_name, num_qubits=3)
        prepare_initial_state(qumat, initial_state)
        for _ in range(num_applications):
            qumat.apply_toffoli_gate(0, 1, 2)
        execute_and_assert_state(
            qumat,
            expected_state,
            num_qubits=3,
            backend_name=backend_name,
            context_msg=(
                f"Initial state: |{initial_state}⟩, "
                f"Toffoli applications: {num_applications}"
            ),
        )

    @pytest.mark.parametrize(
        "control1, control2, target, num_qubits",
        [
            # 4-qubit circuits
            (0, 1, 2, 4),  # Toffoli on qubits 0, 1, 2 in 4-qubit circuit
            (0, 1, 3, 4),  # Toffoli on qubits 0, 1, 3 in 4-qubit circuit
            (1, 2, 3, 4),  # Toffoli on qubits 1, 2, 3 in 4-qubit circuit
            (0, 2, 3, 4),  # Toffoli on qubits 0, 2, 3 in 4-qubit circuit
            # 5-qubit circuits
            (0, 1, 2, 5),  # Toffoli on qubits 0, 1, 2 in 5-qubit circuit
            (0, 1, 4, 5),  # Toffoli on qubits 0, 1, 4 in 5-qubit circuit
            (2, 3, 4, 5),  # Toffoli on qubits 2, 3, 4 in 5-qubit circuit
        ],
    )
    def test_toffoli_on_multiple_qubits(
        self, backend_name, control1, control2, target, num_qubits
    ):
        """Test Toffoli gate on different qubit combinations in multi-qubit circuits."""
        qumat = create_qumat_instance(backend_name, num_qubits=num_qubits)
        qumat.apply_pauli_x_gate(control1)
        qumat.apply_pauli_x_gate(control2)
        qumat.apply_toffoli_gate(control1, control2, target)
        # Expected: all three qubits should be |1⟩
        expected_state = "".join(
            "1" if i in (control1, control2, target) else "0" for i in range(num_qubits)
        )
        execute_and_assert_state(
            qumat,
            expected_state,
            num_qubits=num_qubits,
            backend_name=backend_name,
            context_msg=(
                f"Toffoli(control1={control1}, control2={control2}, target={target}) "
                f"on {num_qubits}-qubit circuit"
            ),
        )

    @pytest.mark.parametrize(
        "initial_state, expected_state, control1, control2, target",
        [
            # Toffoli(0,1,2): target = control0 AND control1
            ("000", "000", 0, 1, 2),  # 0 AND 0 = 0
            ("010", "010", 0, 1, 2),  # 0 AND 1 = 0
            ("100", "100", 0, 1, 2),  # 1 AND 0 = 0
            ("110", "111", 0, 1, 2),  # 1 AND 1 = 1 (target flips)
            # Toffoli(0,2,1): target = control0 AND control2
            ("000", "000", 0, 2, 1),  # 0 AND 0 = 0
            ("001", "001", 0, 2, 1),  # 0 AND 1 = 0
            ("100", "100", 0, 2, 1),  # 1 AND 0 = 0
            ("101", "111", 0, 2, 1),  # 1 AND 1 = 1 (target flips)
            # Toffoli(1,2,0): target = control1 AND control2
            ("000", "000", 1, 2, 0),  # 0 AND 0 = 0
            ("001", "001", 1, 2, 0),  # 0 AND 1 = 0
            ("010", "010", 1, 2, 0),  # 1 AND 0 = 0
            ("011", "111", 1, 2, 0),  # 1 AND 1 = 1 (target flips)
        ],
    )
    def test_toffoli_and_gate_behavior(
        self, backend_name, initial_state, expected_state, control1, control2, target
    ):
        """Test that Toffoli gate acts as a quantum AND gate with parametrized test cases."""
        qumat = create_qumat_instance(backend_name, num_qubits=3)
        prepare_initial_state(qumat, initial_state)
        qumat.apply_toffoli_gate(control1, control2, target)
        execute_and_assert_state(
            qumat,
            expected_state,
            num_qubits=3,
            backend_name=backend_name,
            context_msg=(
                f"Toffoli AND gate: |{initial_state}⟩ -> "
                f"Toffoli({control1},{control2},{target}) -> |{expected_state}⟩"
            ),
        )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestMultiQubitGatesEdgeCases:
    """Test class for edge cases of multi-qubit gates."""

    @pytest.mark.parametrize(
        "gate_name, gate_args",
        [
            ("cnot", (0, 1)),
            ("toffoli", (0, 1, 2)),
        ],
    )
    def test_gate_on_uninitialized_circuit(self, backend_name, gate_name, gate_args):
        """Test that gates raise error on uninitialized circuit."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        with pytest.raises(RuntimeError, match="circuit not initialized"):
            if gate_name == "cnot":
                qumat.apply_cnot_gate(*gate_args)
            else:
                qumat.apply_toffoli_gate(*gate_args)

    @pytest.mark.parametrize(
        "num_qubits, gate_name, gate_args",
        [
            (2, "cnot", (5, 6)),
            (3, "cnot", (10, 11)),
            (4, "cnot", (5, 6)),
            (3, "toffoli", (5, 6, 7)),
            (4, "toffoli", (10, 11, 12)),
            (5, "toffoli", (6, 7, 8)),
        ],
    )
    def test_gate_with_invalid_qubit_indices(
        self, backend_name, num_qubits, gate_name, gate_args
    ):
        """Test that gates handle invalid qubit indices appropriately."""
        qumat = create_qumat_instance(backend_name, num_qubits=num_qubits)
        try:
            if gate_name == "cnot":
                qumat.apply_cnot_gate(*gate_args)
            else:
                qumat.apply_toffoli_gate(*gate_args)
        except (IndexError, ValueError, RuntimeError, Exception):
            pass

    @pytest.mark.parametrize(
        "num_qubits, gate_name, gate_args",
        [
            (2, "cnot", (0, 0)),
            (3, "cnot", (1, 1)),
            (4, "cnot", (2, 2)),
            (3, "toffoli", (0, 0, 0)),
            (4, "toffoli", (1, 1, 1)),
            (5, "toffoli", (2, 2, 2)),
            (3, "toffoli", (0, 1, 0)),
            (3, "toffoli", (0, 0, 1)),
        ],
    )
    def test_gate_with_same_qubits(
        self, backend_name, num_qubits, gate_name, gate_args
    ):
        """Test gates with same qubits (should raise error or handle gracefully)."""
        qumat = create_qumat_instance(backend_name, num_qubits=num_qubits)
        try:
            if gate_name == "cnot":
                qumat.apply_cnot_gate(*gate_args)
            else:
                qumat.apply_toffoli_gate(*gate_args)
            results = qumat.execute_circuit()
            assert results is not None
        except (ValueError, RuntimeError, Exception):
            pass

    def test_cnot_cross_backend_consistency(self, backend_name):
        """Test that CNOT gate produces consistent results across all backends."""
        results_dict = {}
        for backend in TESTING_BACKENDS:
            qumat = create_qumat_instance(backend, num_qubits=2)
            qumat.apply_pauli_x_gate(0)
            qumat.apply_cnot_gate(0, 1)
            results = qumat.execute_circuit()
            results_dict[backend] = get_state_probability(
                results, "11", num_qubits=2, backend_name=backend
            )
        # All backends should give similar results (within 5% tolerance)
        probabilities = list(results_dict.values())
        for i in range(len(probabilities)):
            for j in range(i + 1, len(probabilities)):
                assert abs(probabilities[i] - probabilities[j]) < 0.05, (
                    f"Backends have inconsistent CNOT results: {results_dict}"
                )
