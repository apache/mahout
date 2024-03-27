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
