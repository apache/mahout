# Import the QuantumComputer class from your package
from qumat import QuMat

# Create an instance of QuantumComputer with a specific backend configuration
backend_config = {
    'backend_name': 'qiskit_simulator',  # Replace with the actual backend you want to use
    'backend_options': {
        'simulator_type': 'qasm_simulator',
        'shots': 1024  # Number of shots for measurement
    }
}
qumat = QuMat(backend_config)

# Create a quantum circuit
quantum_circuit = qumat.create_empty_circuit(num_qubits=2)

# Apply quantum gates to the circuit
quantum_circuit.apply_hadamard_gate(qubit_index=0)
quantum_circuit.apply_cnot_gate(control_qubit_index=0, target_qubit_index=1)
quantum_circuit.apply_pauli_x_gate(qubit_index=0)

# Measure the quantum circuit
measurement_results = quantum_circuit.measure()

# Display the measurement results
print("Measurement Results:", measurement_results)
