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

# Import the QuantumComputer class from your package
from qumat import QuMat

# Create an instance of QuantumComputer with a specific backend configuration
backend_config = {
    "backend_name": "qiskit",  # Replace with the actual backend you want to use
    "backend_options": {
        "simulator_type": "aer_simulator",
        "shots": 1024,  # Number of shots for measurement
    },
}
qumat = QuMat(backend_config)

# Create a quantum circuit
qumat.create_empty_circuit(num_qubits=2)

# Apply quantum gates to the circuit
qumat.apply_hadamard_gate(qubit_index=0)
qumat.apply_cnot_gate(control_qubit_index=0, target_qubit_index=1)
qumat.apply_pauli_x_gate(qubit_index=0)
qumat.execute_circuit()
