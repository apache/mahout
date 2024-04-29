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

# Import necessary Qiskit libraries
from qiskit import Aer, QuantumCircuit, execute
from qiskit.quantum_info import Statevector

def get_qumat_backend_config(test_type: str = "get_final_state_vector"):
    if test_type == "get_final_state_vector":
        print("success")
        qumat_backend_config = {
            'backend_name': 'qiskit',
            'backend_options': {
                'simulator_type': 'statevector_simulator',
                'shots': 1
            }
        }
    else:
        pass

    return qumat_backend_config


def get_native_example_final_state_vector(initial_state_ket_str: str = "000") -> Statevector:
    n_qubits = len(initial_state_ket_str)
    assert n_qubits == 3, print("The current qiskit native testing example is strictly 3 qubits")

    simulator = Aer.get_backend('statevector_simulator')

    qc = QuantumCircuit(n_qubits)

    initial_state = Statevector.from_label(initial_state_ket_str)
    qc.initialize(initial_state, range(n_qubits))

    # Create entanglement between qubits 1 and 2
    qc.h(1)  # Apply Hadamard gate on qubit 1
    qc.cx(1, 2)  # Apply CNOT gate with qubit 1 as control and qubit 2 as target

    # Prepare the state to be teleported on qubit 0
    qc.h(0)  # Apply Hadamard gate on qubit 0
    qc.z(0)  # Apply Pauli-Z gate on qubit 0

    # Perform Bell measurement on qubits 0 and 1
    qc.cx(0, 1)  # Apply CNOT gate with qubit 0 as control and qubit 1 as target
    qc.h(0)  # Apply Hadamard gate on qubit 0

    # Simulate the circuit
    job = execute(qc, simulator)
    result = job.result()

    # Get the state vector
    state_vector = result.get_statevector()

    return state_vector
