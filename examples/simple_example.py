# Import the QuantumComputer class from your package
from qumat import QuMat

# Create an instance of QuantumComputer with a specific backend configuration
backend_config = {
    'backend_name': 'qiskit',  # Replace with the actual backend you want to use
    'backend_options': {
        'simulator_type': 'aer_simulator',
        'shots': 1024  # Number of shots for measurement
    }
}
qumat = QuMat(backend_config)

# Create a quantum circuit
qumat.create_empty_circuit(num_qubits=2)

# Apply quantum gates to the circuit
qumat.apply_hadamard_gate(qubit_index=0)
qumat.apply_cnot_gate(control_qubit_index=0, target_qubit_index=1)
qumat.apply_pauli_x_gate(qubit_index=0)
qumat.execute_circuit()
