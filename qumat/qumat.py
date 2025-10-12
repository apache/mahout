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
        self.backend_name = backend_config["backend_name"]
        self.backend_module = import_module(
            f".{self.backend_name}_backend", package="qumat"
        )
        self.backend = self.backend_module.initialize_backend(backend_config)
        self.circuit = None
        self.parameters = {}

    def create_empty_circuit(self, num_qubits):
        self.circuit = self.backend_module.create_empty_circuit(num_qubits)

    def apply_not_gate(self, qubit_index):
        self.backend_module.apply_not_gate(self.circuit, qubit_index)

    def apply_hadamard_gate(self, qubit_index):
        self.backend_module.apply_hadamard_gate(self.circuit, qubit_index)

    def apply_cnot_gate(self, control_qubit_index, target_qubit_index):
        self.backend_module.apply_cnot_gate(
            self.circuit, control_qubit_index, target_qubit_index
        )

    def apply_toffoli_gate(
        self, control_qubit_index1, control_qubit_index2, target_qubit_index
    ):
        self.backend_module.apply_toffoli_gate(
            self.circuit, control_qubit_index1, control_qubit_index2, target_qubit_index
        )

    def apply_swap_gate(self, qubit_index1, qubit_index2):
        self.backend_module.apply_swap_gate(self.circuit, qubit_index1, qubit_index2)

    def apply_cswap_gate(
        self, control_qubit_index, target_qubit_index1, target_qubit_index2
    ):
        self.backend_module.apply_cswap_gate(
            self.circuit, control_qubit_index, target_qubit_index1, target_qubit_index2
        )

    def apply_pauli_x_gate(self, qubit_index):
        self.backend_module.apply_pauli_x_gate(self.circuit, qubit_index)

    def apply_pauli_y_gate(self, qubit_index):
        self.backend_module.apply_pauli_y_gate(self.circuit, qubit_index)

    def apply_pauli_z_gate(self, qubit_index):
        self.backend_module.apply_pauli_z_gate(self.circuit, qubit_index)

    def execute_circuit(self, parameter_values=None):
        if parameter_values:
            self.bind_parameters(parameter_values)
        self.backend_config["parameter_values"] = self.parameters  # Pass parameters
        return self.backend_module.execute_circuit(
            self.circuit, self.backend, self.backend_config
        )

    def bind_parameters(self, parameter_values):
        for param, value in parameter_values.items():
            if param in self.parameters:
                self.parameters[param] = value

    # placeholder method for use in the testing suite
    def get_final_state_vector(self):
        return self.backend_module.get_final_state_vector(
            self.circuit, self.backend, self.backend_config
        )

    def draw(self):
        return self.backend_module.draw_circuit(self.circuit)

    def apply_rx_gate(self, qubit_index, angle):
        self._handle_parameter(angle)
        self.backend_module.apply_rx_gate(self.circuit, qubit_index, angle)

    def apply_ry_gate(self, qubit_index, angle):
        self._handle_parameter(angle)
        self.backend_module.apply_ry_gate(self.circuit, qubit_index, angle)

    def apply_rz_gate(self, qubit_index, angle):
        self._handle_parameter(angle)
        self.backend_module.apply_rz_gate(self.circuit, qubit_index, angle)

    def _handle_parameter(self, param_name):
        if isinstance(param_name, str) and param_name not in self.parameters:
            self.parameters[param_name] = None

    def apply_u_gate(self, qubit_index, theta, phi, lambd):
        self.backend_module.apply_u_gate(self.circuit, qubit_index, theta, phi, lambd)

    def swap_test(self, ancilla_qubit, qubit1, qubit2):
        """
        Implements the swap test circuit for measuring overlap between two quantum states.

        The swap test measures the inner product between the states on qubit1 and qubit2.
        The probability of measuring the ancilla qubit in state |0> is related to the overlap
        as: P(0) = (1 + |<ψ|φ>|²) / 2

        Args:
            ancilla_qubit: Index of the ancilla qubit (should be initialized to |0>)
            qubit1: Index of the first qubit containing state |ψ>
            qubit2: Index of the second qubit containing state |φ>
        """
        # Apply Hadamard to ancilla qubit
        self.apply_hadamard_gate(ancilla_qubit)

        # Apply controlled-SWAP (Fredkin gate) with ancilla as control
        self.apply_cswap_gate(ancilla_qubit, qubit1, qubit2)

        # Apply Hadamard to ancilla qubit again
        self.apply_hadamard_gate(ancilla_qubit)
