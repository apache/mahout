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


class TestParameterBinding:
    """Regression tests for parameter binding functionality across all backends.

    These tests ensure that parameter binding support in all backends
    (Qiskit, Cirq, Amazon Braket) is not accidentally removed or broken.
    """

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_rx_gate_parameter_binding(self, backend_name):
        """Test RX gate parameter binding across all backends."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply parameterized RX gate
        qumat.apply_rx_gate(0, "theta")

        # Execute with parameter binding
        results = qumat.execute_circuit(parameter_values={"theta": math.pi})

        # RX(π) should flip |0⟩ to |1⟩
        prob = get_state_probability(results, "1", num_qubits=1)
        assert prob > 0.95, (
            f"Expected |1⟩ after RX(π) with parameter binding in {backend_name}, "
            f"got probability {prob:.4f}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_ry_gate_parameter_binding(self, backend_name):
        """Test RY gate parameter binding across all backends."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply parameterized RY gate
        qumat.apply_ry_gate(0, "phi")

        # Execute with parameter binding
        results = qumat.execute_circuit(parameter_values={"phi": math.pi / 2})

        # RY(π/2) creates superposition
        # Handle both string and integer state formats (Cirq uses integers)
        if isinstance(results, list):
            results = results[0]

        total_shots = sum(results.values())
        zero_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "0":
                    zero_count = count
                    break
            else:
                if state == 0:
                    zero_count = count
                    break

        prob_zero = zero_count / total_shots if total_shots > 0 else 0.0

        assert 0.45 < prob_zero < 0.55, (
            f"Expected ~0.5 probability for |0⟩ after RY(π/2) in {backend_name}, "
            f"got {prob_zero:.4f}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_rz_gate_parameter_binding(self, backend_name):
        """Test RZ gate parameter binding across all backends."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply parameterized RZ gate
        qumat.apply_rz_gate(0, "lambda")

        # Execute with parameter binding
        results = qumat.execute_circuit(parameter_values={"lambda": math.pi})

        # RZ(π) doesn't change |0⟩ measurement probability
        prob = get_state_probability(results, "0", num_qubits=1)
        assert prob > 0.95, (
            f"Expected |0⟩ after RZ(π) with parameter binding in {backend_name}, "
            f"got probability {prob:.4f}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_multiple_parameter_binding(self, backend_name):
        """Test binding multiple parameters across all backends."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=2)

        # Apply different parameterized gates
        qumat.apply_rx_gate(0, "theta0")
        qumat.apply_ry_gate(1, "phi1")

        # Execute with multiple parameter bindings
        results = qumat.execute_circuit(
            parameter_values={"theta0": math.pi, "phi1": math.pi / 2}
        )

        # Qubit 0 should be |1⟩ (RX(π) = X)
        # Check that we get states with qubit 0 = |1⟩
        # Handle backend-specific result formats
        if isinstance(results, list):
            results = results[0]

        total_shots = sum(results.values())
        target_count = 0

        for state, count in results.items():
            if isinstance(state, str):
                # Qiskit: little-endian (rightmost bit is qubit 0)
                # Amazon Braket: big-endian (leftmost bit is qubit 0)
                if backend_name == "qiskit":
                    # For Qiskit, qubit 0 is rightmost, so check last character
                    if len(state) > 0 and state[-1] == "1":
                        target_count += count
                else:
                    # For Amazon Braket, qubit 0 is leftmost, so check first character
                    if len(state) > 0 and state[0] == "1":
                        target_count += count
            else:
                # Cirq: integer format, big-endian
                # Qubit i is at bit position (num_qubits - 1 - i)
                # For qubit 0 with 2 qubits: bit_position = 2 - 1 - 0 = 1
                num_qubits = 2
                bit_position = num_qubits - 1 - 0
                if ((state >> bit_position) & 1) == 1:
                    target_count += count

        prob = target_count / total_shots if total_shots > 0 else 0.0

        assert prob > 0.4, (
            f"Expected high probability for states with qubit 0=|1⟩ in {backend_name}, "
            f"got {prob:.4f}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_unbound_parameter_error(self, backend_name):
        """Test that unbound parameters raise clear error message across all backends."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)

        # Apply parameterized gate but don't bind
        qumat.apply_rx_gate(0, "theta")

        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="unbound parameters"):
            qumat.execute_circuit()

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_partially_bound_parameters_error(self, backend_name):
        """Test that partially bound parameters raise an error across all backends."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=2)

        # Apply multiple parameterized gates
        qumat.apply_rx_gate(0, "theta0")
        qumat.apply_ry_gate(1, "phi1")

        # Bind only one parameter, leaving the other unbound
        # This should raise ValueError
        with pytest.raises(ValueError, match="unbound parameters"):
            qumat.execute_circuit(parameter_values={"theta0": math.pi})

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_execute_circuit_does_not_mutate_backend_config(self, backend_name):
        """Test that execute_circuit does not mutate the user's backend_config across all backends."""
        backend_config = get_backend_config(backend_name).copy()
        original_config = backend_config.copy()

        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)
        qumat.apply_rx_gate(0, "theta")
        qumat.execute_circuit(parameter_values={"theta": math.pi})

        assert backend_config == original_config, (
            f"backend_config was mutated in {backend_name}; "
            "parameter_values or other keys must not be added."
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_get_final_state_vector_does_not_mutate_backend_config(self, backend_name):
        """Test that get_final_state_vector does not mutate the user's backend_config across all backends."""
        backend_config = get_backend_config(backend_name).copy()
        original_config = backend_config.copy()

        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)
        qumat.apply_rx_gate(0, "theta")
        qumat.bind_parameters({"theta": math.pi / 2})
        qumat.get_final_state_vector()

        assert backend_config == original_config, (
            f"backend_config was mutated by get_final_state_vector in {backend_name}."
        )
