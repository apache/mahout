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
import numpy as np

from .conftest import TESTING_BACKENDS
from qumat import QuMat


class TestOverlapMeasurement:
    """Test overlap measurement functionality across different backends."""

    def get_backend_config(self, backend_name):
        """Get backend configuration by name."""
        configs = {
            "qiskit": {
                "backend_name": "qiskit",
                "backend_options": {
                    "simulator_type": "aer_simulator",
                    "shots": 10000,
                },
            },
            "cirq": {
                "backend_name": "cirq",
                "backend_options": {
                    "simulator_type": "default",
                    "shots": 10000,
                },
            },
            "amazon_braket": {
                "backend_name": "amazon_braket",
                "backend_options": {
                    "simulator_type": "local",
                    "shots": 10000,
                },
            },
        }
        return configs.get(backend_name)

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_identical_zero_states(self, backend_name):
        """Test overlap measurement with two identical |0> states."""
        qumat = QuMat(self.get_backend_config(backend_name))
        qumat.create_empty_circuit(num_qubits=3)
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
        assert overlap > 0.95

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_identical_one_states(self, backend_name):
        """Test overlap measurement with two identical |1> states."""
        qumat = QuMat(self.get_backend_config(backend_name))
        qumat.create_empty_circuit(num_qubits=3)
        qumat.apply_pauli_x_gate(1)
        qumat.apply_pauli_x_gate(2)
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
        assert overlap > 0.95

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_orthogonal_states(self, backend_name):
        """Test overlap measurement with orthogonal states |0> and |1>."""
        qumat = QuMat(self.get_backend_config(backend_name))
        qumat.create_empty_circuit(num_qubits=3)
        qumat.apply_pauli_x_gate(2)
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
        assert overlap < 0.05

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_identical_plus_states(self, backend_name):
        """Test overlap measurement with two identical |+> states."""
        qumat = QuMat(self.get_backend_config(backend_name))
        qumat.create_empty_circuit(num_qubits=3)
        qumat.apply_hadamard_gate(1)
        qumat.apply_hadamard_gate(2)
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
        assert overlap > 0.95

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_plus_minus_states(self, backend_name):
        """Test overlap measurement with |+> and |-> states."""
        qumat = QuMat(self.get_backend_config(backend_name))
        qumat.create_empty_circuit(num_qubits=3)
        qumat.apply_hadamard_gate(1)
        qumat.apply_pauli_x_gate(2)
        qumat.apply_hadamard_gate(2)
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
        assert overlap < 0.05

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_partial_overlap_states(self, backend_name):
        """Test overlap measurement with states having partial overlap."""
        qumat = QuMat(self.get_backend_config(backend_name))
        qumat.create_empty_circuit(num_qubits=3)
        qumat.apply_hadamard_gate(1)
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
        assert 0.4 < overlap < 0.6

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_rotated_states(self, backend_name):
        """Test overlap measurement with rotated states."""
        qumat = QuMat(self.get_backend_config(backend_name))
        qumat.create_empty_circuit(num_qubits=3)
        qumat.apply_ry_gate(1, np.pi / 4)
        qumat.apply_ry_gate(2, np.pi / 4)
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
        assert overlap > 0.95

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_different_rotated_states(self, backend_name):
        """Test overlap measurement with differently rotated states."""
        qumat = QuMat(self.get_backend_config(backend_name))
        qumat.create_empty_circuit(num_qubits=3)
        qumat.apply_ry_gate(1, np.pi / 4)
        qumat.apply_ry_gate(2, np.pi / 2)
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
        expected_overlap = np.cos(np.pi / 8) ** 2
        assert abs(overlap - expected_overlap) < 0.05

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_entangled_states_same(self, backend_name):
        """Test overlap measurement with identical entangled states (Bell states)."""
        qumat = QuMat(self.get_backend_config(backend_name))
        qumat.create_empty_circuit(num_qubits=5)
        qumat.apply_hadamard_gate(1)
        qumat.apply_cnot_gate(1, 2)
        qumat.apply_hadamard_gate(3)
        qumat.apply_cnot_gate(3, 4)
        overlap = qumat.measure_overlap(qubit1=1, qubit2=3, ancilla_qubit=0)
        assert 0.0 <= overlap <= 1.0

    def test_all_backends_consistency(self, testing_backends):
        """Test that all backends produce consistent results."""
        results = {}
        for backend_name in testing_backends:
            qumat = QuMat(self.get_backend_config(backend_name))
            qumat.create_empty_circuit(num_qubits=3)
            results[backend_name] = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        overlaps = list(results.values())
        for i in range(len(overlaps)):
            for j in range(i + 1, len(overlaps)):
                assert abs(overlaps[i] - overlaps[j]) < 0.05

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_measure_overlap_with_different_ancilla(self, backend_name):
        """Test overlap measurement with different ancilla qubit positions."""
        backend_config = self.get_backend_config(backend_name)

        qumat1 = QuMat(backend_config)
        qumat1.create_empty_circuit(num_qubits=4)
        qumat1.apply_hadamard_gate(1)
        qumat1.apply_hadamard_gate(2)
        overlap1 = qumat1.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        qumat2 = QuMat(backend_config)
        qumat2.create_empty_circuit(num_qubits=4)
        qumat2.apply_hadamard_gate(0)
        qumat2.apply_hadamard_gate(1)
        overlap2 = qumat2.measure_overlap(qubit1=0, qubit2=1, ancilla_qubit=3)

        assert overlap1 > 0.95
        assert overlap2 > 0.95
        assert abs(overlap1 - overlap2) < 0.05
