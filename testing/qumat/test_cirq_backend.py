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

from unittest.mock import MagicMock

import cirq
import numpy as np

from qumat.cirq_backend import (
    apply_hadamard_gate,
    apply_rx_gate,
    create_empty_circuit,
    execute_circuit,
    get_final_state_vector,
)


def _hadamard_circuit(num_qubits):
    circuit = create_empty_circuit(num_qubits)
    for i in range(num_qubits):
        apply_hadamard_gate(circuit, i)
    return circuit


def _parameterized_circuit():
    circuit = create_empty_circuit(1)
    apply_rx_gate(circuit, 0, "theta")
    return circuit


# execute_circuit


def test_execute_circuit_uses_given_backend():
    """execute_circuit runs on the passed-in simulator, not a new one."""
    backend = MagicMock(wraps=cirq.Simulator())
    config = {"backend_options": {"shots": 10}}

    result = execute_circuit(_hadamard_circuit(1), backend, config)

    backend.run.assert_called_once()
    assert sum(result[0].values()) == 10


def test_execute_circuit_with_parameters_uses_given_backend():
    """The parameterized path (run_sweep) also uses the passed-in simulator."""
    backend = MagicMock(wraps=cirq.Simulator())
    config = {"backend_options": {"shots": 10}, "parameter_values": {"theta": 0.5}}

    result = execute_circuit(_parameterized_circuit(), backend, config)

    backend.run_sweep.assert_called_once()
    assert sum(result[0].values()) == 10


def test_execute_circuit_seeded_backend_is_reproducible():
    """Two simulators with the same seed give identical histograms."""
    # 3 qubits x 100 shots: unseeded runs essentially never produce equal histograms
    config = {"backend_options": {"shots": 100}}

    first = execute_circuit(_hadamard_circuit(3), cirq.Simulator(seed=1508), config)
    second = execute_circuit(_hadamard_circuit(3), cirq.Simulator(seed=1508), config)

    assert first == second


# get_final_state_vector


def test_get_final_state_vector_uses_given_backend():
    """get_final_state_vector simulates on the passed-in simulator."""
    backend = MagicMock(wraps=cirq.Simulator())

    state = get_final_state_vector(
        _hadamard_circuit(1), backend, {"backend_options": {}}
    )

    backend.simulate.assert_called_once()
    np.testing.assert_allclose(np.abs(state) ** 2, [0.5, 0.5], atol=1e-6)


def test_get_final_state_vector_with_parameters_uses_given_backend():
    """The parameterized simulate path also uses the passed-in simulator."""
    backend = MagicMock(wraps=cirq.Simulator())
    config = {"backend_options": {}, "parameter_values": {"theta": 0.5}}

    get_final_state_vector(_parameterized_circuit(), backend, config)

    backend.simulate.assert_called_once()
