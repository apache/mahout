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
from importlib import import_module

class QuMat:
    def __init__(self, backend_config):
        self.backend_config = backend_config
        self.backend_name = backend_config['backend_name']
        # Dynamically load the backend module based on the user's choice
        self.backend_module = import_module(f".{self.backend_name}_backend", package="qumat")
        self.backend = self.backend_module.initialize_backend(backend_config)
        self.circuit = None

    def create_empty_circuit(self, num_qubits):
        self.circuit = self.backend_module.create_empty_circuit(num_qubits)

    def apply_not_gate(self, qubit_index):
        self.backend_module.apply_not_gate(self.circuit, qubit_index)

    def apply_hadamard_gate(self, qubit_index):
        self.backend_module.apply_hadamard_gate(self.circuit, qubit_index)

    def apply_cnot_gate(self, control_qubit_index, target_qubit_index):
        self.backend_module.apply_cnot_gate(self.circuit, control_qubit_index, target_qubit_index)

    def apply_toffoli_gate(self, control_qubit_index1, control_qubit_index2, target_qubit_index):
        self.backend_module.apply_toffoli_gate(self.circuit, control_qubit_index1, control_qubit_index2, target_qubit_index)

    def apply_swap_gate(self, qubit_index1, qubit_index2):
        self.backend_module.apply_swap_gate(self.circuit, qubit_index1, qubit_index2)

    def apply_pauli_x_gate(self, qubit_index):
        self.backend_module.apply_pauli_x_gate(self.circuit, qubit_index)

    def apply_pauli_y_gate(self, qubit_index):
        self.backend_module.apply_pauli_y_gate(self.circuit, qubit_index)

    def apply_pauli_z_gate(self, qubit_index):
        self.backend_module.apply_pauli_z_gate(self.circuit, qubit_index)

    def execute_circuit(self):
        return self.backend_module.execute_circuit(self.circuit, self.backend, self.backend_config)
