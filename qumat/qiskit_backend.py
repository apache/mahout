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
import qiskit

def initialize_backend(backend_config):
    backend_options = backend_config['backend_options']
    simulator_type = backend_options['simulator_type']
    shots = backend_options['shots']
    backend = qiskit.Aer.get_backend(simulator_type)
    backend.shots = shots
    return backend


def create_empty_circuit(num_qubits):
    return qiskit.QuantumCircuit(num_qubits)

def apply_not_gate(circuit, qubit_index):
    # Apply a NOT gate (X gate) on the specified qubit
    circuit.x(qubit_index)

def apply_hadamard_gate(circuit, qubit_index):
    # Apply a Hadamard gate on the specified qubit
    circuit.h(qubit_index)

def apply_cnot_gate(circuit, control_qubit_index, target_qubit_index):
    # Apply a CNOT gate (controlled-X gate) with the specified control and
    # target qubits
    circuit.cx(control_qubit_index, target_qubit_index)

def apply_toffoli_gate(circuit, control_qubit_index1,
                       control_qubit_index2,
                       target_qubit_index):
    # Apply a Toffoli gate (controlled-controlled-X gate) with the
    # specified control and target qubits
    circuit.ccx(control_qubit_index1,
                     control_qubit_index2,
                     target_qubit_index)

def apply_swap_gate(circuit, qubit_index1, qubit_index2):
    # Apply a SWAP gate to exchange the states of two qubits
    circuit.swap(qubit_index1, qubit_index2)

def apply_pauli_x_gate(circuit, qubit_index):
    # Apply a Pauli X gate on the specified qubit
    circuit.x(qubit_index)

def apply_pauli_y_gate(circuit, qubit_index):
    # Apply a Pauli Y gate on the specified qubit
    circuit.y(qubit_index)

def apply_pauli_z_gate(circuit, qubit_index):
    # Apply a Pauli Z gate on the specified qubit
    circuit.z(qubit_index)

def execute_circuit(circuit, backend, backend_config):
    # Transpile and execute the quantum circuit using the chosen backend
    transpiled_circuit = circuit.transpile(backend)
    job = qiskit.execute(transpiled_circuit, backend,
                         shots=backend_config['backend_options']['shots'])
    result = job.result()
    return result.get_counts(transpiled_circuit)
