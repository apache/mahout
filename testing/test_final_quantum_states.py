from qumat_helpers import get_qumat_example_final_state_vector
import numpy as np
from importlib import import_module

def test_final_state_vector():
    # Specify initial computational basis state vector
    initial_ket_str = "001"

    backends_to_test = ["qiskit"]
    for backend_name in backends_to_test:
        backend_module = import_module(f"{backend_name}_helpers", package="qumat")
        # use native implementation
        native_example_vector = backend_module.get_native_example_final_state_vector(initial_ket_str)

        # use qumat implementation
        qumat_backend_config = backend_module.get_qumat_backend_config("get_final_state_vector")
        qumat_example_vector = get_qumat_example_final_state_vector(qumat_backend_config, initial_ket_str)

        # Compare final state vectors from qumat vs. native implementation
        np.testing.assert_array_equal(qumat_example_vector, native_example_vector)

