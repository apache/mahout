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

from .utils import TESTING_BACKENDS, get_backend_config
from qumat import QuMat


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestPauliXGate:
    """Test class for Pauli X gate functionality."""

    def test_pauli_x_flips_zero_to_one(self, backend_name):
        """Test that Pauli X gate flips |0⟩ to |1⟩."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)
        # Apply Pauli X gate
        qumat.apply_pauli_x_gate(0)
        # Execute circuit
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]
        total_shots = sum(results.values())
        # Count measurements of |1⟩ state
        one_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "1":
                    one_count = count
            else:
                if state == 1:
                    one_count = count
        # Should measure |1⟩ with high probability
        prob_one = one_count / total_shots
        assert prob_one > 0.95, (
            f"Expected |1⟩ state after Pauli X, got probability {prob_one}"
        )

    def test_pauli_x_flips_one_to_zero(self, backend_name):
        """Test that Pauli X gate flips |1⟩ back to |0⟩."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)
        # Prepare |1⟩ state
        qumat.apply_pauli_x_gate(0)
        # Apply Pauli X again to flip back to |0⟩
        qumat.apply_pauli_x_gate(0)
        # Execute circuit
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]
        total_shots = sum(results.values())
        # Count measurements of |0⟩ state
        zero_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "0":
                    zero_count = count
            else:
                if state == 0:
                    zero_count = count
        # Should measure |0⟩ with high probability
        prob_zero = zero_count / total_shots
        assert prob_zero > 0.95, (
            f"Expected |0⟩ state after double Pauli X, got probability {prob_zero}"
        )

    def test_pauli_x_on_multiple_qubits(self, backend_name):
        """Test Pauli X gate on multiple qubits."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=3)
        # Apply Pauli X to qubits 0 and 2
        qumat.apply_pauli_x_gate(0)
        qumat.apply_pauli_x_gate(2)
        # Execute circuit
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]
        total_shots = sum(results.values())

        # Should measure |101⟩ state (qubits 0 and 2 are |1⟩, qubit 1 is |0⟩)
        target_state_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "101":
                    target_state_count = count
            else:
                # For integer format, |101⟩ = 5
                if state == 5:
                    target_state_count = count
        prob_target = target_state_count / total_shots
        assert prob_target > 0.95, (
            f"Expected |101⟩ state, got probability {prob_target}"
        )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestPauliYGate:
    """Test class for Pauli Y gate functionality."""

    def test_pauli_y_on_zero_state(self, backend_name):
        """Test that Pauli Y gate transforms |0⟩ to i|1⟩."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)
        # Apply Pauli Y gate
        qumat.apply_pauli_y_gate(0)
        # Execute circuit
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]
        total_shots = sum(results.values())
        # Count measurements of |1⟩ state
        one_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "1":
                    one_count = count
            else:
                if state == 1:
                    one_count = count
        # Should measure |1⟩ with high probability (phase doesn't affect measurement)
        prob_one = one_count / total_shots
        assert prob_one > 0.95, (
            f"Expected |1⟩ state after Pauli Y, got probability {prob_one}"
        )

    def test_pauli_y_on_one_state(self, backend_name):
        """Test that Pauli Y gate transforms |1⟩ to -i|0⟩."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)
        # Prepare |1⟩ state
        qumat.apply_pauli_x_gate(0)
        # Apply Pauli Y gate
        qumat.apply_pauli_y_gate(0)
        # Execute circuit
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]
        total_shots = sum(results.values())
        # Count measurements of |0⟩ state
        zero_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "0":
                    zero_count = count
            else:
                if state == 0:
                    zero_count = count
        # Should measure |0⟩ with high probability
        prob_zero = zero_count / total_shots
        assert prob_zero > 0.95, (
            f"Expected |0⟩ state after Pauli Y on |1⟩, got probability {prob_zero}"
        )

    def test_pauli_y_double_application(self, backend_name):
        """Test that applying Pauli Y twice returns to original state."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)
        # Apply Pauli Y twice (should return to |0⟩)
        qumat.apply_pauli_y_gate(0)
        qumat.apply_pauli_y_gate(0)
        # Execute circuit
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]
        total_shots = sum(results.values())
        # Count measurements of |0⟩ state
        zero_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "0":
                    zero_count = count
            else:
                if state == 0:
                    zero_count = count
        # Should measure |0⟩ with high probability
        prob_zero = zero_count / total_shots
        assert prob_zero > 0.95, (
            f"Expected |0⟩ state after double Pauli Y, got probability {prob_zero}"
        )


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestPauliZGate:
    """Test class for Pauli Z gate functionality."""

    def test_pauli_z_on_zero_state(self, backend_name):
        """Test that Pauli Z gate leaves |0⟩ unchanged."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)
        # Apply Pauli Z gate
        qumat.apply_pauli_z_gate(0)
        # Execute circuit
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]

        total_shots = sum(results.values())
        # Count measurements of |0⟩ state
        zero_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "0":
                    zero_count = count
            else:
                if state == 0:
                    zero_count = count
        # Should measure |0⟩ with high probability
        prob_zero = zero_count / total_shots
        assert prob_zero > 0.95, (
            f"Expected |0⟩ state after Pauli Z, got probability {prob_zero}"
        )

    def test_pauli_z_on_one_state(self, backend_name):
        """Test that Pauli Z gate flips phase of |1⟩ to -|1⟩."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=1)
        # Prepare |1⟩ state
        qumat.apply_pauli_x_gate(0)
        # Apply Pauli Z gate
        qumat.apply_pauli_z_gate(0)
        # Execute circuit
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]
        total_shots = sum(results.values())
        # Count measurements of |1⟩ state (phase flip doesn't affect measurement)
        one_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "1":
                    one_count = count
            else:
                if state == 1:
                    one_count = count
        # Should still measure |1⟩ with high probability
        prob_one = one_count / total_shots
        assert prob_one > 0.95, (
            f"Expected |1⟩ state after Pauli Z on |1⟩, got probability {prob_one}"
        )
