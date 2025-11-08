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
from importlib import import_module


class QuMat:
    """Unified interface for quantum circuit operations across multiple backends.

    Provides a consistent API for creating and manipulating quantum circuits
    using different quantum computing backends (Qiskit, Cirq, Amazon Braket).
    Abstracts backend-specific details for gate operations, circuit execution,
    and state measurement.

    :param backend_config: Configuration dictionary for the quantum backend.
        Must contain ``backend_name`` (str) and ``backend_options`` (dict).
        The ``backend_options`` should include ``simulator_type`` and ``shots``.
    :type backend_config: dict
    """

    def __init__(self, backend_config):
        """Create a QuMat instance with the specified backend configuration.

        :param backend_config: Configuration dictionary containing backend name
            and options. Required keys:
            - ``backend_name``: Name of the backend (e.g., "qiskit", "cirq", "amazon_braket")
            - ``backend_options``: Dictionary with backend-specific options
        :type backend_config: dict
        :raises ImportError: If the specified backend module cannot be imported.
        :raises KeyError: If required configuration keys are missing.
        """
        self.backend_config = backend_config
        self.backend_name = backend_config["backend_name"]
        self.backend_module = import_module(
            f".{self.backend_name}_backend", package="qumat"
        )
        self.backend = self.backend_module.initialize_backend(backend_config)
        self.circuit = None
        self.num_qubits = None
        self.parameters = {}

    def create_empty_circuit(self, num_qubits: int | None = None):
        """Create an empty quantum circuit with the specified number of qubits.

        Must be called before applying any gates or executing operations.

        :param num_qubits: Number of qubits in the circuit. If ``None``,
            creates a circuit without pre-allocated qubits.
        :type num_qubits: int | None, optional
        """
        self.num_qubits = num_qubits
        self.circuit = self.backend_module.create_empty_circuit(num_qubits)

    def _ensure_circuit_initialized(self):
        """Ensure the circuit has been created before operations.

        Checks if the circuit has been initialized via ``create_empty_circuit()``.
        Raises ``RuntimeError`` if not initialized.

        :raises RuntimeError: If the circuit has not been initialized.
        """
        if self.circuit is None:
            raise RuntimeError(
                "circuit not initialized. call create_empty_circuit(num_qubits) "
                "before applying gates or executing operations."
            )

    def apply_not_gate(self, qubit_index):
        """Apply a NOT gate (Pauli-X gate) to the specified qubit.

        Flips the qubit state from |0⟩ to |1⟩ or |1⟩ to |0⟩.
        Equivalent to the Pauli-X gate.

        :param qubit_index: Index of the qubit.
        :type qubit_index: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_not_gate(self.circuit, qubit_index)

    def apply_hadamard_gate(self, qubit_index):
        """Apply a Hadamard gate to the specified qubit.

        Creates a superposition state, transforming |0⟩ to (|0⟩ + |1⟩)/√2
        and |1⟩ to (|0⟩ - |1⟩)/√2.

        :param qubit_index: Index of the qubit.
        :type qubit_index: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_hadamard_gate(self.circuit, qubit_index)

    def apply_cnot_gate(self, control_qubit_index, target_qubit_index):
        """Apply a Controlled-NOT (CNOT) gate between two qubits.

        Fundamental for entangling qubits. Flips the target qubit if and only
        if the control qubit is in the |1⟩ state.

        :param control_qubit_index: Index of the control qubit.
        :type control_qubit_index: int
        :param target_qubit_index: Index of the target qubit.
        :type target_qubit_index: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_cnot_gate(
            self.circuit, control_qubit_index, target_qubit_index
        )

    def apply_toffoli_gate(
        self, control_qubit_index1, control_qubit_index2, target_qubit_index
    ):
        """Apply a Toffoli gate (CCX gate) to three qubits.

        Acts as a quantum AND gate. Flips the target qubit if and only if
        both control qubits are in the |1⟩ state.

        :param control_qubit_index1: Index of the first control qubit.
        :type control_qubit_index1: int
        :param control_qubit_index2: Index of the second control qubit.
        :type control_qubit_index2: int
        :param target_qubit_index: Index of the target qubit.
        :type target_qubit_index: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_toffoli_gate(
            self.circuit, control_qubit_index1, control_qubit_index2, target_qubit_index
        )

    def apply_swap_gate(self, qubit_index1, qubit_index2):
        """Swap the states of two qubits.

        :param qubit_index1: Index of the first qubit.
        :type qubit_index1: int
        :param qubit_index2: Index of the second qubit.
        :type qubit_index2: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_swap_gate(self.circuit, qubit_index1, qubit_index2)

    def apply_cswap_gate(
        self, control_qubit_index, target_qubit_index1, target_qubit_index2
    ):
        """Apply a controlled-SWAP (Fredkin) gate.

        Swaps the states of two target qubits if and only if the control
        qubit is in the |1⟩ state.

        :param control_qubit_index: Index of the control qubit.
        :type control_qubit_index: int
        :param target_qubit_index1: Index of the first target qubit.
        :type target_qubit_index1: int
        :param target_qubit_index2: Index of the second target qubit.
        :type target_qubit_index2: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_cswap_gate(
            self.circuit, control_qubit_index, target_qubit_index1, target_qubit_index2
        )

    def apply_pauli_x_gate(self, qubit_index):
        """Apply a Pauli-X gate to the specified qubit.

        Equivalent to the NOT gate. Flips the qubit state from |0⟩ to |1⟩
        or |1⟩ to |0⟩.

        :param qubit_index: Index of the qubit.
        :type qubit_index: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_pauli_x_gate(self.circuit, qubit_index)

    def apply_pauli_y_gate(self, qubit_index):
        """Apply a Pauli-Y gate to the specified qubit.

        Rotates the qubit around the Y-axis of the Bloch sphere, affecting
        both phase and amplitude.

        :param qubit_index: Index of the qubit.
        :type qubit_index: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_pauli_y_gate(self.circuit, qubit_index)

    def apply_pauli_z_gate(self, qubit_index):
        """Apply a Pauli-Z gate to the specified qubit.

        Rotates the qubit around the Z-axis of the Bloch sphere, altering
        the phase without changing the amplitude.

        :param qubit_index: Index of the qubit.
        :type qubit_index: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_pauli_z_gate(self.circuit, qubit_index)

    def execute_circuit(self, parameter_values=None):
        """Execute the quantum circuit and return the measurement results.

        Runs the circuit on the configured backend. For parameterized circuits,
        provide parameter values to bind before execution.

        :param parameter_values: Dictionary mapping parameter names to numerical
            values. Binds these values to circuit parameters before execution.
        :type parameter_values: dict, optional
        :returns: Measurement results. Format depends on the backend:
            - Qiskit/Braket: Dictionary with state strings as keys and counts as values
            - Cirq: List of dictionaries with integer states as keys
        :rtype: dict | list[dict]
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        if self.num_qubits == 0:
            shots = self.backend_config["backend_options"].get("shots", 1)
            if self.backend_name == "cirq":
                return [{0: shots}]
            else:
                return {"": shots}

        if parameter_values:
            self.bind_parameters(parameter_values)
        self.backend_config["parameter_values"] = self.parameters  # Pass parameters
        return self.backend_module.execute_circuit(
            self.circuit, self.backend, self.backend_config
        )

    def bind_parameters(self, parameter_values):
        """Bind numerical values to circuit parameters.

        Assigns numerical values to symbolic parameters defined in parameterized
        gates.

        :param parameter_values: Dictionary mapping parameter names to numerical
            values.
        :type parameter_values: dict
        :raises ValueError: If a parameter name is not found in the circuit's
            parameter list.
        """
        for param, value in parameter_values.items():
            if param not in self.parameters:
                raise ValueError(
                    f"parameter '{param}' not found in circuit. "
                    f"available parameters: {list(self.parameters.keys())}"
                )
            self.parameters[param] = value

    def get_final_state_vector(self):
        """Return the final state vector of the quantum circuit.

        The complete quantum state vector after circuit execution,
        representing the full quantum state of all qubits.

        :returns: The final state vector as a numpy array.
        :rtype: numpy.ndarray
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        return self.backend_module.get_final_state_vector(
            self.circuit, self.backend, self.backend_config
        )

    def draw(self):
        """Visualize the quantum circuit.

        Generates a visual representation of the circuit. The output format
        depends on the backend implementation.

        :returns: Circuit visualization. The exact type depends on the backend.
        :rtype: str | object
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        return self.backend_module.draw_circuit(self.circuit)

    def apply_rx_gate(self, qubit_index, angle):
        """Apply a rotation around the X-axis to the specified qubit.

        Rotates the qubit by the given angle around the X-axis of the Bloch
        sphere. The angle can be a static value or a parameter name for
        parameterized circuits.

        :param qubit_index: Index of the qubit.
        :type qubit_index: int
        :param angle: Rotation angle in radians. Can be a float or a string
            parameter name.
        :type angle: float | str
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self._handle_parameter(angle)
        self.backend_module.apply_rx_gate(self.circuit, qubit_index, angle)

    def apply_ry_gate(self, qubit_index, angle):
        """Apply a rotation around the Y-axis to the specified qubit.

        Rotates the qubit by the given angle around the Y-axis of the Bloch
        sphere. The angle can be a static value or a parameter name for
        parameterized circuits.

        :param qubit_index: Index of the qubit.
        :type qubit_index: int
        :param angle: Rotation angle in radians. Can be a float or a string
            parameter name.
        :type angle: float | str
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self._handle_parameter(angle)
        self.backend_module.apply_ry_gate(self.circuit, qubit_index, angle)

    def apply_rz_gate(self, qubit_index, angle):
        """Apply a rotation around the Z-axis to the specified qubit.

        Rotates the qubit by the given angle around the Z-axis of the Bloch
        sphere. The angle can be a static value or a parameter name for
        parameterized circuits.

        :param qubit_index: Index of the qubit.
        :type qubit_index: int
        :param angle: Rotation angle in radians. Can be a float or a string
            parameter name.
        :type angle: float | str
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self._handle_parameter(angle)
        self.backend_module.apply_rz_gate(self.circuit, qubit_index, angle)

    def _handle_parameter(self, param_name):
        """Register parameter names when parameterized gates are applied.

        Automatically adds string parameter names to the parameters dictionary
        if not already registered.

        :param param_name: Parameter name to handle. If it's a string,
            registers it as a parameter.
        :type param_name: str | float
        """
        if isinstance(param_name, str) and param_name not in self.parameters:
            self.parameters[param_name] = None

    def apply_u_gate(self, qubit_index, theta, phi, lambd):
        """Apply a U gate (universal single-qubit gate) to the specified qubit.

        A universal single-qubit gate parameterized by three angles (theta,
        phi, lambd) that can represent any single-qubit unitary operation.

        :param qubit_index: Index of the qubit.
        :type qubit_index: int
        :param theta: First rotation angle in radians.
        :type theta: float
        :param phi: Second rotation angle in radians.
        :type phi: float
        :param lambd: Third rotation angle in radians.
        :type lambd: float
        :raises RuntimeError: If the circuit has not been initialized.
        """
        self._ensure_circuit_initialized()
        self.backend_module.apply_u_gate(self.circuit, qubit_index, theta, phi, lambd)

    def swap_test(self, ancilla_qubit, qubit1, qubit2):
        """Implement the swap test circuit for measuring overlap between two quantum states.

        Measures the inner product between the states on ``qubit1`` and ``qubit2``.
        The probability of measuring the ancilla qubit in state |0⟩ is related
        to the overlap as: P(0) = (1 + |⟨ψ|φ⟩|²) / 2

        :param ancilla_qubit: Index of the ancilla qubit (should be initialized to |0⟩).
        :type ancilla_qubit: int
        :param qubit1: Index of the first qubit containing state |ψ⟩.
        :type qubit1: int
        :param qubit2: Index of the second qubit containing state |φ⟩.
        :type qubit2: int
        :raises RuntimeError: If the circuit has not been initialized.
        """
        # Apply Hadamard to ancilla qubit
        self.apply_hadamard_gate(ancilla_qubit)

        # Apply controlled-SWAP (Fredkin gate) with ancilla as control
        self.apply_cswap_gate(ancilla_qubit, qubit1, qubit2)

        # Apply Hadamard to ancilla qubit again
        self.apply_hadamard_gate(ancilla_qubit)

    def measure_overlap(self, qubit1, qubit2, ancilla_qubit=0):
        """Measure the overlap (fidelity) between two quantum states using the swap test.

        Creates a swap test circuit to calculate the similarity between the
        quantum states on ``qubit1`` and ``qubit2``. Returns the squared overlap
        |⟨ψ|φ⟩|², which represents the fidelity between the two states.

        The swap test measures P(ancilla=0), related to overlap as:
        P(0) = (1 + |⟨ψ|φ⟩|²) / 2

        For certain states (especially identical excited states), global phase
        effects may cause the ancilla to measure predominantly |1⟩ instead of |0⟩.
        This method handles both cases by taking the measurement probability
        closer to 1.

        :param qubit1: Index of the first qubit containing state |ψ⟩.
        :type qubit1: int
        :param qubit2: Index of the second qubit containing state |φ⟩.
        :type qubit2: int
        :param ancilla_qubit: Index of the ancilla qubit. Default is 0. Should be
            initialized to |0⟩.
        :type ancilla_qubit: int, optional
        :returns: The squared overlap |⟨ψ|φ⟩|² between the two states (fidelity),
            clamped to the range [0.0, 1.0].
        :rtype: float
        :raises RuntimeError: If the circuit has not been initialized.
        """
        # Perform the swap test
        self.swap_test(ancilla_qubit, qubit1, qubit2)
        results = self.execute_circuit()

        # Calculate the probability of measuring ancilla in |0> state
        prob_zero = self.calculate_prob_zero(results, ancilla_qubit)
        prob_zero_or_one = max(prob_zero, 1 - prob_zero)
        overlap_squared = 2 * prob_zero_or_one - 1
        overlap_squared = max(0.0, min(1.0, overlap_squared))

        return overlap_squared

    def calculate_prob_zero(self, results, ancilla_qubit):
        """Calculate the probability of measuring the ancilla qubit in |0⟩ state.

        Delegates to the backend-specific implementation. Different backends
        may use different qubit ordering conventions (little-endian vs big-endian).

        :param results: Measurement results from ``execute_circuit()``. Format
            depends on the backend.
        :type results: dict | list[dict]
        :param ancilla_qubit: Index of the ancilla qubit.
        :type ancilla_qubit: int
        :returns: Probability of measuring the ancilla qubit in |0⟩ state.
        :rtype: float
        """
        return self.backend_module.calculate_prob_zero(
            results, ancilla_qubit, self.num_qubits
        )
