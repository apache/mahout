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

import cirq
import numpy as np
import sympy


def initialize_backend(backend_config: dict[str, Any]) -> cirq.Simulator:
    # Assuming 'simulator_type' specifies the type of simulator in Cirq
    simulator_type = backend_config.get("backend_options", {}).get(
        "simulator_type", "default"
    )
    if simulator_type != "default":
        print(
            f"Simulator type '{simulator_type}' is not supported in Cirq. Ignoring this argument"
        )

    return cirq.Simulator()


def create_empty_circuit(num_qubits: int | None = None) -> cirq.Circuit:
    circuit = cirq.Circuit()
    if num_qubits is not None:
        qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
        for qubit in qubits:
            circuit.append(cirq.I(qubit))
    return circuit


def apply_not_gate(circuit: cirq.Circuit, qubit_index: int) -> None:
    qubit = cirq.LineQubit(qubit_index)
    circuit.append(cirq.X(qubit))


def apply_hadamard_gate(circuit: cirq.Circuit, qubit_index: int) -> None:
    qubit = cirq.LineQubit(qubit_index)
    circuit.append(cirq.H(qubit))


def apply_cnot_gate(
    circuit: cirq.Circuit, control_qubit_index: int, target_qubit_index: int
) -> None:
    control_qubit = cirq.LineQubit(control_qubit_index)
    target_qubit = cirq.LineQubit(target_qubit_index)
    circuit.append(cirq.CNOT(control_qubit, target_qubit))


def apply_toffoli_gate(
    circuit: cirq.Circuit,
    control_qubit_index1: int,
    control_qubit_index2: int,
    target_qubit_index: int,
) -> None:
    control_qubit1 = cirq.LineQubit(control_qubit_index1)
    control_qubit2 = cirq.LineQubit(control_qubit_index2)
    target_qubit = cirq.LineQubit(target_qubit_index)
    circuit.append(cirq.CCX(control_qubit1, control_qubit2, target_qubit))


def apply_swap_gate(circuit: cirq.Circuit, qubit_index1: int, qubit_index2: int) -> None:
    qubit1 = cirq.LineQubit(qubit_index1)
    qubit2 = cirq.LineQubit(qubit_index2)
    circuit.append(cirq.SWAP(qubit1, qubit2))


def apply_cswap_gate(
    circuit: cirq.Circuit,
    control_qubit_index: int,
    target_qubit_index1: int,
    target_qubit_index2: int,
) -> None:
    control_qubit = cirq.LineQubit(control_qubit_index)
    target_qubit1 = cirq.LineQubit(target_qubit_index1)
    target_qubit2 = cirq.LineQubit(target_qubit_index2)
    circuit.append(cirq.CSWAP(control_qubit, target_qubit1, target_qubit2))


def apply_pauli_x_gate(circuit: cirq.Circuit, qubit_index: int) -> None:
    qubit = cirq.LineQubit(qubit_index)
    circuit.append(cirq.X(qubit))


def apply_pauli_y_gate(circuit: cirq.Circuit, qubit_index: int) -> None:
    qubit = cirq.LineQubit(qubit_index)
    circuit.append(cirq.Y(qubit))


def apply_pauli_z_gate(circuit: cirq.Circuit, qubit_index: int) -> None:
    qubit = cirq.LineQubit(qubit_index)
    circuit.append(cirq.Z(qubit))


def execute_circuit(
    circuit: cirq.Circuit, backend: cirq.Simulator, backend_config: dict[str, Any]
) -> list[dict[int, int]]:
    # handle 0-qubit circuits before adding measurements
    if not circuit.all_qubits():
        shots = backend_config["backend_options"].get("shots", 1)
        return [{0: shots}]

    # Ensure measurement is added to capture the results
    if not circuit.has_measurements():
        circuit.append(cirq.measure(*circuit.all_qubits(), key="result"))
    simulator = cirq.Simulator()
    parameter_values = backend_config.get("parameter_values", None)
    if parameter_values:
        # Convert parameter_values to applicable resolvers
        res = [cirq.ParamResolver(parameter_values)]
        results = simulator.run_sweep(
            circuit,
            repetitions=backend_config["backend_options"].get("shots", 1),
            params=res,
        )
        return [result.histogram(key="result") for result in results]
    else:
        result = simulator.run(
            circuit, repetitions=backend_config["backend_options"].get("shots", 1)
        )
        return [result.histogram(key="result")]


def draw_circuit(circuit: cirq.Circuit) -> None:
    print(circuit)


def apply_rx_gate(circuit: cirq.Circuit, qubit_index: int, angle: float | str) -> None:
    param = sympy.Symbol(angle) if isinstance(angle, str) else angle
    qubit = cirq.LineQubit(qubit_index)
    circuit.append(cirq.rx(param).on(qubit))


def apply_ry_gate(circuit: cirq.Circuit, qubit_index: int, angle: float | str) -> None:
    param = sympy.Symbol(angle) if isinstance(angle, str) else angle
    qubit = cirq.LineQubit(qubit_index)
    circuit.append(cirq.ry(param).on(qubit))


def apply_rz_gate(circuit: cirq.Circuit, qubit_index: int, angle: float | str) -> None:
    param = sympy.Symbol(angle) if isinstance(angle, str) else angle
    qubit = cirq.LineQubit(qubit_index)
    circuit.append(cirq.rz(param).on(qubit))


def apply_u_gate(
    circuit: cirq.Circuit, qubit_index: int, theta: float, phi: float, lambd: float
) -> None:
    qubit = cirq.LineQubit(qubit_index)
    circuit.append(cirq.rz(lambd).on(qubit))
    circuit.append(cirq.ry(theta).on(qubit))
    circuit.append(cirq.rz(phi).on(qubit))


def get_final_state_vector(
    circuit: cirq.Circuit, backend: cirq.Simulator, backend_config: dict[str, Any]
) -> np.ndarray:
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    return result.final_state_vector


def calculate_prob_zero(
    results: list[dict[int, int]] | dict[int, int], ancilla_qubit: int, num_qubits: int
) -> float:
    """
    Calculate the probability of measuring the ancilla qubit in |0> state.

    Cirq uses big-endian qubit ordering with integer format results,
    where qubit i corresponds to bit (num_qubits - 1 - i).

    Args:
        results: Measurement results from execute_circuit() (list of dicts with integer keys)
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
        bit_position = num_qubits - 1 - ancilla_qubit
        if ((state >> bit_position) & 1) == 0:
            count_zero += count

    return count_zero / total_shots if total_shots > 0 else 0.0
