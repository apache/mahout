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

from braket.devices import LocalSimulator
from braket.circuits import Circuit
import numpy as np


def get_qumat_backend_config(test_type: str = "get_final_state_vector"):
    if test_type == "get_final_state_vector":
        qumat_backend_config = {
            "backend_name": "amazon_braket",
            "backend_options": {"simulator_type": "local", "shots": 1},
        }
    else:
        pass

    return qumat_backend_config


def get_native_example_final_state_vector(
    initial_state_ket_str: str = "000",
) -> np.ndarray:
    n_qubits = len(initial_state_ket_str)
    assert n_qubits == 3, "The current braket native testing example is strictly 3 qubits"

    # Use LocalSimulator for state vector simulation
    device = LocalSimulator()

    circuit = Circuit()

    # Initialize to desired state
    for i, bit in enumerate(initial_state_ket_str):
        if bit == "1":
            circuit.x(i)

    # Create entanglement between qubits 1 and 2
    circuit.h(1)
    circuit.cnot(1, 2)

    # Prepare the state to be teleported on qubit 0
    circuit.h(0)
    circuit.z(0)

    # Perform Bell measurement on qubits 0 and 1
    circuit.cnot(0, 1)
    circuit.h(0)

    # Add state_vector result type to get the final state
    circuit.state_vector()

    # Run the circuit
    result = device.run(circuit, shots=0).result()

    # Get the state vector (values is a complex numpy array)
    state_vector = result.values[0]

    return state_vector
