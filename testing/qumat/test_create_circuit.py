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

from qumat import QuMat

from ..utils import TESTING_BACKENDS, get_backend_config


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestCreateCircuit:
    """Test class for create_empty_circuit functionality."""

    def test_create_empty_circuit(self, backend_name):
        """Test that create_empty_circuit works"""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=2)

        assert qumat.circuit is not None

    @pytest.mark.parametrize("num_qubits", [None, 1.5, "2", True])
    def test_create_empty_circuit_rejects_non_int_num_qubits(
        self, backend_name, num_qubits
    ):
        """create_empty_circuit raises TypeError when num_qubits is not an int."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        with pytest.raises(TypeError, match="num_qubits must be an integer"):
            qumat.create_empty_circuit(num_qubits)

        assert qumat.circuit is None
        assert qumat.num_qubits is None

    @pytest.mark.parametrize("num_qubits", [-1, -5])
    def test_create_empty_circuit_rejects_negative_num_qubits(
        self, backend_name, num_qubits
    ):
        """create_empty_circuit raises ValueError when num_qubits is negative."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        with pytest.raises(ValueError, match="num_qubits cannot be negative"):
            qumat.create_empty_circuit(num_qubits)

        assert qumat.circuit is None
        assert qumat.num_qubits is None

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


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestDefaultShotsParameter:
    """Test that backends work without explicitly specifying shots parameter."""

    def test_execute_circuit_without_shots_parameter(self, backend_name):
        """Test that circuits can be executed without specifying shots.

        This is a regression test to ensure all backends have a default
        value for the shots parameter, matching the README quick start example.
        """
        # Config without shots - should use default
        configs = {
            "qiskit": {
                "backend_name": "qiskit",
                "backend_options": {"simulator_type": "aer_simulator"},
            },
            "cirq": {
                "backend_name": "cirq",
                "backend_options": {"simulator_type": "default"},
            },
            "amazon_braket": {
                "backend_name": "amazon_braket",
                "backend_options": {"simulator_type": "local"},
            },
        }

        backend_config = configs[backend_name]
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=2)
        qumat.apply_hadamard_gate(0)
        qumat.apply_cnot_gate(0, 1)

        # Should not raise KeyError for missing 'shots'
        result = qumat.execute_circuit()

        assert result is not None


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestCircuitMutationPrevention:
    """Test that circuits are not mutated during execution."""

    def test_execute_circuit_multiple_times_no_mutation(self, backend_name):
        """Test that executing a circuit multiple times doesn't mutate it"""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)
        qumat.create_empty_circuit(num_qubits=2)
        qumat.apply_hadamard_gate(0)
        qumat.apply_cnot_gate(0, 1)

        assert qumat.circuit is not None
        circuit = qumat.circuit

        # Record initial circuit state
        if backend_name == "cirq":
            initial_length = len(circuit)
            initial_has_measurements = circuit.has_measurements()
        elif backend_name == "qiskit":
            initial_length = len(circuit.data)
        else:
            initial_length = len(circuit.instructions)

        # Execute circuit multiple times
        result1 = qumat.execute_circuit()
        result2 = qumat.execute_circuit()
        result3 = qumat.execute_circuit()

        # Verify circuit hasn't been mutated
        assert qumat.circuit is not None
        circuit_after = qumat.circuit
        if backend_name == "cirq":
            final_length = len(circuit_after)
            final_has_measurements = circuit_after.has_measurements()
            assert initial_length == final_length, (
                f"Circuit length changed from {initial_length} to {final_length}. "
                "Circuit was mutated!"
            )
            assert initial_has_measurements == final_has_measurements, (
                "Circuit measurement state changed. Circuit was mutated!"
            )
        elif backend_name == "qiskit":
            final_length = len(circuit_after.data)
            assert initial_length == final_length, (
                f"Circuit length changed from {initial_length} to {final_length}. "
                "Circuit was mutated!"
            )
        else:
            final_length = len(circuit_after.instructions)
            assert initial_length == final_length, (
                f"Circuit length changed from {initial_length} to {final_length}. "
                "Circuit was mutated!"
            )

        # Verify results are consistent (all should have same structure)
        if isinstance(result1, list):
            result1 = result1[0]
        if isinstance(result2, list):
            result2 = result2[0]
        if isinstance(result3, list):
            result3 = result3[0]

        # All results should have the same keys (measurement states)
        assert set(result1.keys()) == set(result2.keys()) == set(result3.keys()), (
            "Results have different states after multiple executions"
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

    def test_qiskit_custom_simulator_fallback(self):
        """Test that Qiskit backend accepts simulator types outside the legacy map."""
        config = {
            "backend_name": "qiskit",
            "backend_options": {"simulator_type": "matrix_product_state"},
        }

        qumat = QuMat(config)

        assert qumat.backend is not None
        assert qumat.backend.options.method == "matrix_product_state"

    @pytest.mark.parametrize(
        ("backend_name", "simulator_type", "supported"),
        [
            ("qiskit", "not_a_method", "'aer_simulator'"),
            ("cirq", "density_matrix", "'default'"),
            ("amazon_braket", "sv1", "'local'"),
        ],
    )
    def test_unsupported_simulator_type_raises(
        self, capsys, backend_name, simulator_type, supported
    ):
        """Every backend rejects an unknown simulator_type the same way (#1507).

        It raises ValueError naming the bad value and the supported ones, and
        prints nothing (the old behaviour printed a notice and carried on).
        """
        config = {
            "backend_name": backend_name,
            "backend_options": {"simulator_type": simulator_type, "shots": 10},
        }

        with pytest.raises(ValueError, match="Unsupported simulator_type") as excinfo:
            QuMat(config)

        message = str(excinfo.value)
        assert repr(simulator_type) in message
        assert supported in message
        assert capsys.readouterr().out == ""

    def test_qiskit_unsupported_simulator_type_chains_aer_error(self):
        """The Qiskit ValueError keeps Aer's error as its cause and lists Aer methods."""
        from qiskit_aer import AerError

        config = {
            "backend_name": "qiskit",
            "backend_options": {"simulator_type": "bogus"},
        }

        with pytest.raises(ValueError) as excinfo:
            QuMat(config)

        assert isinstance(excinfo.value.__cause__, AerError)
        assert "'matrix_product_state'" in str(excinfo.value)

    @pytest.mark.parametrize("backend_options", [{}, {"simulator_type": "default"}])
    def test_cirq_default_simulator_type_accepted(self, capsys, backend_options):
        """Cirq still accepts 'default' or no simulator_type, without printing."""
        qumat = QuMat({"backend_name": "cirq", "backend_options": backend_options})

        assert qumat.backend is not None
        assert capsys.readouterr().out == ""

    def test_braket_local_simulator_type_accepted(self, capsys):
        """'local' still selects Braket's offline simulator, without printing."""
        from braket.devices import LocalSimulator

        qumat = QuMat(
            {
                "backend_name": "amazon_braket",
                "backend_options": {"simulator_type": "local", "shots": 10},
            }
        )

        assert isinstance(qumat.backend, LocalSimulator)
        assert capsys.readouterr().out == ""


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestCircuitDrawing:
    """Test class for circuit drawing functionality."""

    def test_draw_circuit_returns_output(self, backend_name):
        """Test that draw_circuit runs successfully and returns an output."""
        backend_config = get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        qumat.create_empty_circuit(num_qubits=2)
        qumat.apply_hadamard_gate(0)
        qumat.apply_cnot_gate(0, 1)

        drawing = qumat.draw_circuit()

        assert drawing is not None
