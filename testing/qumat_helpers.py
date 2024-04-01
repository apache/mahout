import numpy as np
from functools import reduce
from qumat.qumat import QuMat
from qiskit import execute

class BinaryString(str):
    def __new__(cls, value):
        if not all(char in ['0', '1'] for char in value):
            raise ValueError("String contains characters other than '0' and '1'")
        return str.__new__(cls, value)


def create_np_computational_basis_state(ket_str: BinaryString,
                                        np_dtype: str = "complex128") -> np.array:
    single_qubit_state_dict = {
                            "0": np.array([1, 0], dtype=np_dtype),
                            "1": np.array([0, 1], dtype=np_dtype)
                            }

    single_qubit_vectors = map(single_qubit_state_dict.get, ket_str)
    computational_basis_vector = reduce(np.kron, single_qubit_vectors)

    return computational_basis_vector


def get_qumat_example_final_state_vector(backend_config: dict, initial_state_ket_str: BinaryString = "000"):
    n_qubits = len(initial_state_ket_str)
    assert n_qubits == 3, print("The current qumat testing example is strictly 3 qubits")

    qumat_instance = QuMat(backend_config)

    qumat_instance.create_empty_circuit(num_qubits=3)
    initial_state = create_np_computational_basis_state(initial_state_ket_str)
    qumat_instance.circuit.initialize(initial_state, range(n_qubits))

    qumat_instance.apply_hadamard_gate(qubit_index=1)
    qumat_instance.apply_cnot_gate(control_qubit_index=1, target_qubit_index=2)
    qumat_instance.apply_hadamard_gate(qubit_index=0)
    qumat_instance.apply_pauli_z_gate(qubit_index=0)
    qumat_instance.apply_cnot_gate(control_qubit_index=0, target_qubit_index=1)
    qumat_instance.apply_hadamard_gate(qubit_index=0)

    # Simulate the circuit
    job = execute(qumat_instance.circuit, qumat_instance.backend, shots=1)
    result = job.result()

    # Get the state vector
    state_vector = result.get_statevector()

    return state_vector