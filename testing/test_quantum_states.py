from qumat.qumat import QuMat

# Placeholder test, will delete
def test_instantiation():
    # Create an instance of QuantumComputer with a specific backend configuration
    backend_config = {
        'backend_name': 'qiskit_simulator',  # Replace with the actual backend you want to use
        'backend_options': {
            'simulator_type': 'qasm_simulator',
            'shots': 1024  # Number of shots for measurement
        }
    }
    qumat = QuMat(backend_config)
