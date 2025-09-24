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
import numpy as np
from functools import reduce
from qumat.qumat import QuMat


class BinaryString(str):
    def __new__(cls, value):
        if not all(char in ["0", "1"] for char in value):
            raise ValueError("String contains characters other than '0' and '1'")
        return str.__new__(cls, value)


def create_np_computational_basis_state(
    ket_str: BinaryString, np_dtype: str = "complex128"
) -> np.array:
    single_qubit_state_dict = {
        "0": np.array([1, 0], dtype=np_dtype),
        "1": np.array([0, 1], dtype=np_dtype),
    }

    single_qubit_vectors = map(single_qubit_state_dict.get, ket_str)
    computational_basis_vector = reduce(np.kron, single_qubit_vectors)

    return computational_basis_vector


def get_qumat_example_final_state_vector(
    backend_config: dict, initial_state_ket_str: BinaryString = "000"
):
    n_qubits = len(initial_state_ket_str)
    assert n_qubits == 3, print(
        "The current qumat testing example is strictly 3 qubits"
    )

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

    state_vector = qumat_instance.get_final_state_vector()

    return state_vector
