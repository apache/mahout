import numpy as np
from functools import reduce

class BinaryString(str):
    def __new__(cls, value):
        if not all(char in ['0', '1'] for char in value):
            raise ValueError("String contains characters other than '0' and '1'")
        return str.__new__(cls, value)


# Initialize basis state as numpy array
def create_np_computational_basis_state(ket_str: BinaryString,
                                        np_dtype: str = "complex128") -> np.array:
    single_qubit_state_dict = {
                            "0": np.array([1, 0], dtype=np_dtype),
                            "1": np.array([0, 1], dtype=np_dtype)
                            }

    single_qubit_vectors = map(single_qubit_state_dict.get, ket_str)
    computational_basis_vector = reduce(lambda x, y: np.kron(x, y), single_qubit_vectors)

    return computational_basis_vector