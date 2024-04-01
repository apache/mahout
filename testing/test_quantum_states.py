from qumat_helpers import get_qumat_example_final_state_vector
from qiskit_helpers import get_qiskit_native_example_final_state_vector
import numpy as np

# Placeholder test, will delete
def test_final_state_vector():
    # Prepare data needed for qumat instance
    qumat_backend_config = {
        'backend_name': 'qiskit',
        'backend_options': {
            'simulator_type': 'statevector_simulator',
            'shots': 1
        }
    }
    # Specify initial computational basis state vector
    initial_ket_str = "001"
    # Execute qumat and native implementations
    qumat_example_vector = get_qumat_example_final_state_vector(qumat_backend_config, initial_ket_str)
    qiskit_example_vector = get_qiskit_native_example_final_state_vector(initial_ket_str)

    # Compare final state vectors from qumat vs. native implementation
    np.testing.assert_array_equal(qumat_example_vector, qiskit_example_vector)

