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
class TestCreateCircuit:
    """Test class for create_empty_circuit functionality."""

    def test_create_empty_circuit(self, backend_name):
        """Test that create_empty_circuit works"""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit()

        assert qumat.circuit is not None

    @pytest.mark.parametrize("num_qubits", [0, 1, 3, 5])
    def test_create_circuit_initializes_to_zero(self, backend_name, num_qubits):
        """Test that create_empty_circuit properly initializes all qubits to |0⟩."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with specified number of qubits
        qumat.create_empty_circuit(num_qubits)

        # Execute and verify all qubits measure |0⟩
        results = qumat.execute_circuit()
        if isinstance(results, list):
            results = results[0]

        total_shots = sum(results.values())
        assert total_shots > 0

        # Find count for all-zeros state
        zero_state_count = 0
        for state, count in results.items():
            if isinstance(state, str):
                if state == "0" * num_qubits:
                    zero_state_count = count
            else:
                if state == 0:
                    zero_state_count = count

        assert zero_state_count > 0.95 * total_shots, (
            f"Expected |0...0⟩ state, got {zero_state_count}/{total_shots}"
        )


class TestBackendConfigValidation:
    """Test class for backend configuration validation."""

    @pytest.mark.parametrize("invalid_config", [None, "not a dict"])
    def test_invalid_type_raises_valueerror(self, invalid_config):
        """Test that non-dictionary backend_config raises ValueError."""
        with pytest.raises(ValueError, match="backend_config must be a dictionary"):
            QuMat(invalid_config)

    def test_missing_backend_name_raises_keyerror(self):
        """Test that missing backend_name raises KeyError with helpful message."""
        config = {"backend_options": {"simulator_type": "aer_simulator", "shots": 1024}}

        with pytest.raises(KeyError, match="missing required key.*backend_name"):
            QuMat(config)

    def test_missing_backend_options_raises_keyerror(self):
        """Test that missing backend_options raises KeyError with helpful message."""
        config = {"backend_name": "qiskit"}

        with pytest.raises(KeyError, match="missing required key.*backend_options"):
            QuMat(config)
