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
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from numbers import Real
from typing import Any, SupportsFloat

import boto3
from braket.aws import AwsDevice, AwsSession
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator

CircuitLike = Any
BraketBackend = AwsDevice | LocalSimulator


def initialize_backend(backend_config: Mapping[str, Any]) -> BraketBackend:
    backend_options = backend_config["backend_options"]
    simulator_type = backend_options.get("simulator_type", "default")
    region = backend_options.get("region")

    # Create AWS session with region if specified
    aws_session = None
    if region:
        boto_session = boto3.Session(region_name=region)
        aws_session = AwsSession(boto_session=boto_session)

    if simulator_type == "local":
        return LocalSimulator()
    elif simulator_type == "default":
        return AwsDevice(
            "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            aws_session=aws_session,
        )
    else:
        print(
            f"Simulator type '{simulator_type}' is not supported in Amazon Braket. Using default."
        )
        return AwsDevice(
            "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            aws_session=aws_session,
        )


def _as_angle(angle: Any) -> Any:
    """Normalise a gate angle for Braket.

    Strings name a symbolic parameter. Any real number, including NumPy
    scalars such as ``np.float32`` or ``np.int64``, is converted to a
    plain ``float``: Braket only accepts Python numbers here and would
    otherwise try to turn the value into a ``FreeParameter`` name.
    Anything else (an existing ``FreeParameter`` or expression) passes
    through unchanged.
    """
    if isinstance(angle, str):
        return FreeParameter(angle)
    if isinstance(angle, Real):
        return float(angle)
    return angle


def _bound_inputs(
    parameter_values: Mapping[str, Any], circuit: CircuitLike
) -> dict[str, float]:
    """Map bound parameter values to Braket ``inputs``.

    Only names present in the circuit are passed, and values are cast to
    ``float`` because Braket's simulators reject NumPy scalar types.
    """
    names = {p.name for p in circuit.parameters}
    return {
        name: float(value) for name, value in parameter_values.items() if name in names
    }


def create_empty_circuit(num_qubits: int | None = None) -> CircuitLike:
    circuit: CircuitLike = Circuit()
    if num_qubits is not None:
        for i in range(num_qubits):
            circuit.i(i)  # type: ignore[unresolved-attribute]
    return circuit


def apply_not_gate(circuit: CircuitLike, qubit_index: int) -> None:
    circuit.x(qubit_index)


def apply_hadamard_gate(circuit: CircuitLike, qubit_index: int) -> None:
    circuit.h(qubit_index)


def apply_cnot_gate(
    circuit: CircuitLike, control_qubit_index: int, target_qubit_index: int
) -> None:
    circuit.cnot(control_qubit_index, target_qubit_index)


def apply_toffoli_gate(
    circuit: CircuitLike,
    control_qubit_index1: int,
    control_qubit_index2: int,
    target_qubit_index: int,
) -> None:
    circuit.ccnot(control_qubit_index1, control_qubit_index2, target_qubit_index)


def apply_swap_gate(circuit: CircuitLike, qubit_index1: int, qubit_index2: int) -> None:
    circuit.swap(qubit_index1, qubit_index2)


def apply_cswap_gate(
    circuit: CircuitLike,
    control_qubit_index: int,
    target_qubit_index1: int,
    target_qubit_index2: int,
) -> None:
    circuit.cswap(control_qubit_index, target_qubit_index1, target_qubit_index2)


def apply_pauli_x_gate(circuit: CircuitLike, qubit_index: int) -> None:
    circuit.x(qubit_index)


def apply_pauli_y_gate(circuit: CircuitLike, qubit_index: int) -> None:
    circuit.y(qubit_index)


def apply_pauli_z_gate(circuit: CircuitLike, qubit_index: int) -> None:
    circuit.z(qubit_index)


def apply_t_gate(circuit: CircuitLike, qubit_index: int) -> None:
    circuit.t(qubit_index)


def execute_circuit(
    circuit: CircuitLike,
    backend: BraketBackend,
    backend_config: MutableMapping[str, Any],
) -> dict[str, int]:
    shots = backend_config["backend_options"].get("shots", 1)
    parameter_values = backend_config.get("parameter_values", {})
    if parameter_values and circuit.parameters:
        inputs = _bound_inputs(parameter_values, circuit)
        task = backend.run(circuit, shots=shots, inputs=inputs)
    else:
        task = backend.run(circuit, shots=shots)
    result = task.result()
    return result.measurement_counts  # ty: ignore[possibly-missing-attribute]


# placeholder method for use in the testing suite
def get_final_state_vector(
    circuit: CircuitLike,
    backend: BraketBackend,
    backend_config: MutableMapping[str, Any],
) -> Any:
    # Work on a copy: ``state_vector()`` appends a StateVector result type
    # to the circuit in place, and Braket rejects any later shot-based run
    # of a circuit carrying that result type ("StateVector or Amplitude
    # cannot be specified when shots>0").
    working_circuit = circuit.copy()
    working_circuit.state_vector()
    parameter_values = backend_config.get("parameter_values", {})
    if parameter_values and working_circuit.parameters:
        inputs = _bound_inputs(parameter_values, working_circuit)
        result = backend.run(working_circuit, shots=0, inputs=inputs).result()
    else:
        result = backend.run(working_circuit, shots=0).result()
    state_vector = result.values[0]  # ty: ignore[possibly-missing-attribute]

    return state_vector


def draw_circuit(circuit: CircuitLike) -> str:
    # Unfortunately, Amazon Braket does not have direct support for drawing circuits in the same way
    # as Qiskit and Cirq. You would typically visualize Amazon Braket circuits using external tools.
    # For simplicity, we'll return the circuit object's string representation.
    return str(circuit)


def apply_rx_gate(
    circuit: CircuitLike,
    qubit_index: int,
    angle: float | int | str | FreeParameter,
) -> None:
    circuit.rx(qubit_index, _as_angle(angle))


def apply_ry_gate(
    circuit: CircuitLike,
    qubit_index: int,
    angle: float | int | str | FreeParameter,
) -> None:
    circuit.ry(qubit_index, _as_angle(angle))


def apply_rz_gate(
    circuit: CircuitLike,
    qubit_index: int,
    angle: float | int | str | FreeParameter,
) -> None:
    circuit.rz(qubit_index, _as_angle(angle))


def apply_u_gate(
    circuit: CircuitLike,
    qubit_index: int,
    theta: SupportsFloat,
    phi: SupportsFloat,
    lambd: SupportsFloat,
) -> None:
    # U(θ, φ, λ) = Rz(φ) · Ry(θ) · Rz(λ)
    # U is not parameterisable in Qumat (on any backend), so only numeric
    # normalisation applies here; strings are left for Braket to reject.
    theta, phi, lambd = (
        float(a) if isinstance(a, Real) else a for a in (theta, phi, lambd)
    )
    circuit.rz(qubit_index, lambd)
    circuit.ry(qubit_index, theta)
    circuit.rz(qubit_index, phi)


def calculate_prob_zero(
    results: dict[str, int] | list[dict[str, int]],
    ancilla_qubit: int,
    num_qubits: int,
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
