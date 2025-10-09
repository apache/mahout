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

import cirq
import numpy as np


def get_qumat_backend_config(test_type: str = "get_final_state_vector"):
    if test_type == "get_final_state_vector":
        qumat_backend_config = {
            "backend_name": "cirq",
            "backend_options": {"simulator_type": "default", "shots": 1},
        }
    else:
        pass

    return qumat_backend_config


def get_native_example_final_state_vector(
    initial_state_ket_str: str = "000",
) -> np.ndarray:
    n_qubits = len(initial_state_ket_str)
    assert n_qubits == 3, "The current cirq native testing example is strictly 3 qubits"

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()

    # Initialize to desired state
    for i, bit in enumerate(initial_state_ket_str):
        if bit == "1":
            circuit.append(cirq.X(qubits[i]))

    # Create entanglement between qubits 1 and 2
    circuit.append(cirq.H(qubits[1]))
    circuit.append(cirq.CNOT(qubits[1], qubits[2]))

    # Prepare the state to be teleported on qubit 0
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.Z(qubits[0]))

    # Perform Bell measurement on qubits 0 and 1
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.H(qubits[0]))

    # Simulate the circuit
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)

    return result.final_state_vector
