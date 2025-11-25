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
from typing import Any

import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator


def initialize_backend(backend_config: dict[str, Any]) -> LocalSimulator | AwsDevice:
    backend_options = backend_config["backend_options"]
    simulator_type = backend_options.get("simulator_type", "default")
    if simulator_type == "local":
        return LocalSimulator()
    elif simulator_type == "default":
        return AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    else:
        print(
            f"Simulator type '{simulator_type}' is not supported in Amazon Braket. Using default."
        )
        return AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")


def create_empty_circuit(num_qubits: int | None = None) -> Circuit:
    circuit = Circuit()
    if num_qubits is not None:
        for i in range(num_qubits):
            circuit.i(i)
    return circuit


def apply_not_gate(circuit: Circuit, qubit_index: int) -> None:
    circuit.x(qubit_index)


def apply_hadamard_gate(circuit: Circuit, qubit_index: int) -> None:
    circuit.h(qubit_index)


def apply_cnot_gate(circuit: Circuit, control_qubit_index: int, target_qubit_index: int) -> None:
    circuit.cnot(control_qubit_index, target_qubit_index)


def apply_toffoli_gate(
    circuit: Circuit,
    control_qubit_index1: int,
    control_qubit_index2: int,
    target_qubit_index: int,
) -> None:
    circuit.ccnot(control_qubit_index1, control_qubit_index2, target_qubit_index)


def apply_swap_gate(circuit: Circuit, qubit_index1: int, qubit_index2: int) -> None:
    circuit.swap(qubit_index1, qubit_index2)


def apply_cswap_gate(
    circuit: Circuit,
    control_qubit_index: int,
    target_qubit_index1: int,
    target_qubit_index2: int,
) -> None:
    circuit.cswap(control_qubit_index, target_qubit_index1, target_qubit_index2)


def apply_pauli_x_gate(circuit: Circuit, qubit_index: int) -> None:
    circuit.x(qubit_index)


def apply_pauli_y_gate(circuit: Circuit, qubit_index: int) -> None:
    circuit.y(qubit_index)


def apply_pauli_z_gate(circuit: Circuit, qubit_index: int) -> None:
    circuit.z(qubit_index)


def execute_circuit(
    circuit: Circuit, backend: LocalSimulator | AwsDevice, backend_config: dict[str, Any]
) -> dict[str, int]:
    shots = backend_config["backend_options"].get("shots", 1)
    parameter_values = backend_config.get("parameter_values", {})
    if parameter_values and circuit.parameters:
        # Braket accepts parameter names as strings in inputs dict
        inputs = {
            param_name: value
            for param_name, value in parameter_values.items()
            if param_name in {p.name for p in circuit.parameters}
        }
        task = backend.run(circuit, shots=shots, inputs=inputs)
    else:
        task = backend.run(circuit, shots=shots)
    result = task.result()
    return result.measurement_counts


# placeholder method for use in the testing suite
def get_final_state_vector(
    circuit: Circuit, backend: LocalSimulator | AwsDevice, backend_config: dict[str, Any]
) -> np.ndarray:
    circuit.state_vector()
    result = backend.run(circuit, shots=0).result()
    state_vector = result.values[0]

    return state_vector


def draw_circuit(circuit: Circuit) -> None:
    # Unfortunately, Amazon Braket does not have direct support for drawing circuits in the same way
    # as Qiskit and Cirq. You would typically visualize Amazon Braket circuits using external tools.
    # For simplicity, we'll print the circuit object which gives some textual representation.
    print(circuit)


def apply_rx_gate(circuit: Circuit, qubit_index: int, angle: float | str) -> None:
    if isinstance(angle, (int, float)):
        circuit.rx(qubit_index, angle)
    else:
        param = FreeParameter(angle)
        circuit.rx(qubit_index, param)


def apply_ry_gate(circuit: Circuit, qubit_index: int, angle: float | str) -> None:
    if isinstance(angle, (int, float)):
        circuit.ry(qubit_index, angle)
    else:
        param = FreeParameter(angle)
        circuit.ry(qubit_index, param)


def apply_rz_gate(circuit: Circuit, qubit_index: int, angle: float | str) -> None:
    if isinstance(angle, (int, float)):
        circuit.rz(qubit_index, angle)
    else:
        param = FreeParameter(angle)
        circuit.rz(qubit_index, param)


def apply_u_gate(
    circuit: Circuit, qubit_index: int, theta: float, phi: float, lambd: float
) -> None:
    # U(θ, φ, λ) = Rz(φ) · Ry(θ) · Rz(λ)
    circuit.rz(qubit_index, lambd)
    circuit.ry(qubit_index, theta)
    circuit.rz(qubit_index, phi)


def calculate_prob_zero(
    results: dict[str, int] | list[dict[str, int]], ancilla_qubit: int, num_qubits: int
) -> float:
    """
    Calculate the probability of measuring the ancilla qubit in |0> state.

    Amazon Braket uses big-endian qubit ordering with string format results,
    where the leftmost bit corresponds to qubit 0.

    Args:
        results: Measurement results from execute_circuit() (dict with string keys)
        ancilla_qubit: Index of the ancilla qubit
        num_qubits: Total number of qubits in the circuit

    Returns:
        float: Probability of measuring ancilla in |0> state
    """
    if isinstance(results, list):
        results = results[0]

    total_shots = sum(results.values())
    count_zero = 0

    for state, count in results.items():
        # Braket: big-endian, leftmost bit is qubit 0
        if len(state) > ancilla_qubit and state[ancilla_qubit] == "0":
            count_zero += count

    return count_zero / total_shots if total_shots > 0 else 0.0
