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
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit, FreeParameter


def initialize_backend(backend_config):
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


def create_empty_circuit(num_qubits):
    return Circuit()


def apply_not_gate(circuit, qubit_index):
    circuit.x(qubit_index)


def apply_hadamard_gate(circuit, qubit_index):
    circuit.h(qubit_index)


def apply_cnot_gate(circuit, control_qubit_index, target_qubit_index):
    circuit.cnot(control_qubit_index, target_qubit_index)


def apply_toffoli_gate(
    circuit, control_qubit_index1, control_qubit_index2, target_qubit_index
):
    circuit.ccnot(control_qubit_index1, control_qubit_index2, target_qubit_index)


def apply_swap_gate(circuit, qubit_index1, qubit_index2):
    circuit.swap(qubit_index1, qubit_index2)


def apply_cswap_gate(
    circuit, control_qubit_index, target_qubit_index1, target_qubit_index2
):
    circuit.cswap(control_qubit_index, target_qubit_index1, target_qubit_index2)


def apply_pauli_x_gate(circuit, qubit_index):
    circuit.x(qubit_index)


def apply_pauli_y_gate(circuit, qubit_index):
    circuit.y(qubit_index)


def apply_pauli_z_gate(circuit, qubit_index):
    circuit.z(qubit_index)


def execute_circuit(circuit, backend, backend_config):
    shots = backend_config["backend_options"].get("shots", 1)
    task = backend.run(circuit, shots=shots)
    result = task.result()
    return result.measurement_counts


# placeholder method for use in the testing suite
def get_final_state_vector(circuit, backend, backend_config):
    circuit.state_vector()
    result = backend.run(circuit, shots=0).result()
    state_vector = result.values[0]

    return state_vector


def draw_circuit(circuit):
    # Unfortunately, Amazon Braket does not have direct support for drawing circuits in the same way
    # as Qiskit and Cirq. You would typically visualize Amazon Braket circuits using external tools.
    # For simplicity, we'll print the circuit object which gives some textual representation.
    print(circuit)


def apply_rx_gate(circuit, qubit_index, angle):
    if isinstance(angle, (int, float)):
        circuit.rx(qubit_index, angle)
    else:
        param = FreeParameter(angle)
        circuit.rx(qubit_index, param)


def apply_ry_gate(circuit, qubit_index, angle):
    if isinstance(angle, (int, float)):
        circuit.ry(qubit_index, angle)
    else:
        param = FreeParameter(angle)
        circuit.ry(qubit_index, param)


def apply_rz_gate(circuit, qubit_index, angle):
    if isinstance(angle, (int, float)):
        circuit.rz(qubit_index, angle)
    else:
        param = FreeParameter(angle)
        circuit.rz(qubit_index, param)


def apply_u_gate(circuit, qubit_index, theta, phi, lambd):
    circuit.rx(qubit_index, theta)
    circuit.ry(qubit_index, phi)
    circuit.rz(qubit_index, lambd)
