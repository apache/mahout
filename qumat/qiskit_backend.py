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
import qiskit
from qiskit_aer import Aer, AerSimulator
from typing import Any, cast
import numpy as np


def initialize_backend(backend_config: dict[str, Any]) -> Any:
    """Initialize the Qiskit backend with the specified configuration.

    Args:
        backend_config: Configuration dictionary containing backend options.
            Must include 'backend_options' with 'simulator_type' and 'shots'.

    Returns:
        Configured Qiskit backend instance.
    """
    backend_options = backend_config["backend_options"]
    simulator_type = backend_options["simulator_type"]
    shots = backend_options["shots"]
    backend = Aer.get_backend(simulator_type)
    backend.shots = shots
    return backend


def create_empty_circuit(num_qubits: int | None = None) -> qiskit.QuantumCircuit:
    """Create an empty quantum circuit with optional qubit allocation.

    Args:
        num_qubits: Number of qubits to allocate. If None, creates circuit
            without pre-allocated qubits.

    Returns:
        Empty Qiskit QuantumCircuit instance.
    """
    if num_qubits is not None:
        return qiskit.QuantumCircuit(num_qubits)
    else:
        return qiskit.QuantumCircuit()


def apply_not_gate(circuit: qiskit.QuantumCircuit, qubit_index: int) -> None:
    """Apply a NOT gate (Pauli-X gate) to the specified qubit.

    Flips the qubit state from |0⟩ to |1⟩ or |1⟩ to |0⟩.

    Args:
        circuit: The quantum circuit to modify.
        qubit_index: Index of the target qubit.
    """
    circuit.x(qubit_index)


def apply_hadamard_gate(circuit: qiskit.QuantumCircuit, qubit_index: int) -> None:
    """Apply a Hadamard gate to create superposition.

    Transforms |0⟩ to (|0⟩ + |1⟩)/√2 and |1⟩ to (|0⟩ - |1⟩)/√2.

    Args:
        circuit: The quantum circuit to modify.
        qubit_index: Index of the target qubit.
    """
    circuit.h(qubit_index)


def apply_cnot_gate(
    circuit: qiskit.QuantumCircuit, control_qubit_index: int, target_qubit_index: int
) -> None:
    """Apply a Controlled-NOT (CNOT) gate for entanglement.

    Flips the target qubit if and only if the control qubit is in |1⟩ state.

    Args:
        circuit: The quantum circuit to modify.
        control_qubit_index: Index of the control qubit.
        target_qubit_index: Index of the target qubit.
    """
    circuit.cx(control_qubit_index, target_qubit_index)


def apply_toffoli_gate(
    circuit: qiskit.QuantumCircuit,
    control_qubit_index1: int,
    control_qubit_index2: int,
    target_qubit_index: int,
) -> None:
    """Apply a Toffoli (CCX) gate - quantum AND operation.

    Flips the target qubit if and only if both control qubits are in |1⟩ state.

    Args:
        circuit: The quantum circuit to modify.
        control_qubit_index1: Index of the first control qubit.
        control_qubit_index2: Index of the second control qubit.
        target_qubit_index: Index of the target qubit.
    """
    circuit.ccx(control_qubit_index1, control_qubit_index2, target_qubit_index)


def apply_swap_gate(
    circuit: qiskit.QuantumCircuit, qubit_index1: int, qubit_index2: int
) -> None:
    """Apply a SWAP gate to exchange states of two qubits.

    Args:
        circuit: The quantum circuit to modify.
        qubit_index1: Index of the first qubit.
        qubit_index2: Index of the second qubit.
    """
    circuit.swap(qubit_index1, qubit_index2)


def apply_cswap_gate(
    circuit: qiskit.QuantumCircuit,
    control_qubit_index: int,
    target_qubit_index1: int,
    target_qubit_index2: int,
) -> None:
    """Apply a controlled-SWAP (Fredkin) gate.

    Swaps the states of two target qubits if the control qubit is in |1⟩ state.

    Args:
        circuit: The quantum circuit to modify.
        control_qubit_index: Index of the control qubit.
        target_qubit_index1: Index of the first target qubit.
        target_qubit_index2: Index of the second target qubit.
    """
    circuit.cswap(control_qubit_index, target_qubit_index1, target_qubit_index2)


def apply_pauli_x_gate(circuit: qiskit.QuantumCircuit, qubit_index: int) -> None:
    """Apply a Pauli-X gate (equivalent to NOT gate).

    Args:
        circuit: The quantum circuit to modify.
        qubit_index: Index of the target qubit.
    """
    circuit.x(qubit_index)


def apply_pauli_y_gate(circuit: qiskit.QuantumCircuit, qubit_index: int) -> None:
    """Apply a Pauli-Y gate for combined bit-flip and phase-flip.

    Args:
        circuit: The quantum circuit to modify.
        qubit_index: Index of the target qubit.
    """
    circuit.y(qubit_index)


def apply_pauli_z_gate(circuit: qiskit.QuantumCircuit, qubit_index: int) -> None:
    """Apply a Pauli-Z gate for phase flip without state flip.

    Args:
        circuit: The quantum circuit to modify.
        qubit_index: Index of the target qubit.
    """
    circuit.z(qubit_index)


def execute_circuit(
    circuit: qiskit.QuantumCircuit, backend: Any, backend_config: dict[str, Any]
) -> dict[str, int]:
    """Execute the quantum circuit and return measurement results.

    Automatically adds measurements if not present, handles parameterized circuits,
    and runs the circuit on the specified backend.

    Args:
        circuit: The quantum circuit to execute.
        backend: Qiskit backend to run the circuit on.
        backend_config: Configuration including shots and parameter values.

    Returns:
        Dictionary mapping measurement outcome strings to counts.
    """
    # Add measurements if they are not already present
    # Check if circuit already has measurement operations
    has_measurements = any(
        isinstance(inst.operation, qiskit.circuit.Measure) for inst in circuit.data
    )
    if not has_measurements:
        circuit.measure_all()

    # Ensure the circuit is parameterized properly
    if circuit.parameters:
        # Parse the global parameter configuration
        parameter_bindings = {
            param: backend_config["parameter_values"][str(param)]
            for param in circuit.parameters
        }
        transpiled_circuit = qiskit.transpile(circuit, backend)
        bound_circuit = transpiled_circuit.assign_parameters(parameter_bindings)
        job = backend.run(
            bound_circuit, shots=backend_config["backend_options"]["shots"]
        )
        result = job.result()
        return cast(dict[str, int], result.get_counts())
    else:
        transpiled_circuit = qiskit.transpile(circuit, backend)
        job = backend.run(
            transpiled_circuit, shots=backend_config["backend_options"]["shots"]
        )
        result = job.result()
        return cast(dict[str, int], result.get_counts())


def get_final_state_vector(
    circuit: qiskit.QuantumCircuit, backend: Any, backend_config: dict[str, Any]
) -> np.ndarray:
    """Get the final state vector of the quantum circuit.

    Used primarily for testing and validation. Returns the complete quantum
    state as a statevector.

    Args:
        circuit: The quantum circuit to simulate.
        backend: Qiskit backend (not used, statevector simulator is created).
        backend_config: Backend configuration options.

    Returns:
        NumPy array representing the final quantum state vector.
    """
    simulator = AerSimulator(method="statevector")

    # Add save_statevector instruction
    circuit.save_statevector()

    # Simulate the circuit
    transpiled_circuit = qiskit.transpile(circuit, simulator)
    job = simulator.run(transpiled_circuit)
    result = job.result()

    return cast(np.ndarray, result.get_statevector())


def draw_circuit(circuit: qiskit.QuantumCircuit) -> None:
    """Visualize the quantum circuit.

    Prints a text-based diagram of the circuit structure.

    Args:
        circuit: The quantum circuit to visualize.
    """
    print(circuit.draw())


def apply_rx_gate(
    circuit: qiskit.QuantumCircuit, qubit_index: int, angle: float | str
) -> None:
    """Apply an X-axis rotation gate.

    Rotates the qubit by the specified angle around the X-axis of the Bloch sphere.

    Args:
        circuit: The quantum circuit to modify.
        qubit_index: Index of the target qubit.
        angle: Rotation angle in radians (float) or parameter name (str).
    """
    param = qiskit.circuit.Parameter(angle) if isinstance(angle, str) else angle
    circuit.rx(param, qubit_index)


def apply_ry_gate(
    circuit: qiskit.QuantumCircuit, qubit_index: int, angle: float | str
) -> None:
    """Apply a Y-axis rotation gate.

    Rotates the qubit by the specified angle around the Y-axis of the Bloch sphere.

    Args:
        circuit: The quantum circuit to modify.
        qubit_index: Index of the target qubit.
        angle: Rotation angle in radians (float) or parameter name (str).
    """
    param = qiskit.circuit.Parameter(angle) if isinstance(angle, str) else angle
    circuit.ry(param, qubit_index)


def apply_rz_gate(
    circuit: qiskit.QuantumCircuit, qubit_index: int, angle: float | str
) -> None:
    """Apply a Z-axis rotation gate.

    Rotates the qubit by the specified angle around the Z-axis of the Bloch sphere.

    Args:
        circuit: The quantum circuit to modify.
        qubit_index: Index of the target qubit.
        angle: Rotation angle in radians (float) or parameter name (str).
    """
    param = qiskit.circuit.Parameter(angle) if isinstance(angle, str) else angle
    circuit.rz(param, qubit_index)


def apply_u_gate(
    circuit: qiskit.QuantumCircuit, qubit_index: int, theta: float, phi: float, lambd: float
) -> None:
    """Apply a universal single-qubit gate.

    The U gate can represent any single-qubit unitary operation using three
    rotation angles.

    Args:
        circuit: The quantum circuit to modify.
        qubit_index: Index of the target qubit.
        theta: First rotation angle in radians.
        phi: Second rotation angle in radians.
        lambd: Third rotation angle in radians.
    """
    # Apply the U gate directly with specified parameters
    circuit.u(theta, phi, lambd, qubit_index)


def calculate_prob_zero(
    results: dict[str, int], ancilla_qubit: int, num_qubits: int
) -> float:
    """
    Calculate the probability of measuring the ancilla qubit in |0> state.

    Qiskit uses little-endian qubit ordering with string format results,
    where the rightmost bit corresponds to qubit 0.

    Args:
        results: Measurement results from execute_circuit() (dict with string keys)
        ancilla_qubit: Index of the ancilla qubit
        num_qubits: Total number of qubits in the circuit

    Returns:
        float: Probability of measuring ancilla in |0> state
    """
    # Handle different result formats from different backends
    if isinstance(results, list):
        results = results[0]

    total_shots = sum(results.values())
    count_zero = 0

    for state, count in results.items():
        # Qiskit: little-endian, rightmost bit is qubit 0
        if len(state) > ancilla_qubit and state[-(ancilla_qubit + 1)] == "0":
            count_zero += count

    return count_zero / total_shots if total_shots > 0 else 0.0
