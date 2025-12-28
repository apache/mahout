# QuMat Class Methods

## `__init__(self, backend_config)`
- **Purpose**: Initializes the QuMat instance with a specific backend.
- **Parameters**:
    - `backend_config` (dict): Configuration for the backend including its name and options.
- **Usage**: Used to set up the quantum computing backend based on user configuration.

## `create_empty_circuit(self, num_qubits)`
- **Purpose**: Creates an empty quantum circuit with a specified number of qubits.
- **Parameters**:
    - `num_qubits` (int): Number of qubits in the quantum circuit.
- **Usage**: Used at the start of quantum computations to prepare a new quantum circuit.

## `apply_not_gate(self, qubit_index)`
- **Purpose**: Applies a NOT gate (quantum equivalent of a classical NOT) to a specified qubit.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit to which the gate is applied.
- **Usage**: Used to flip the state of a qubit from 0 to 1 or from 1 to 0.

## `apply_hadamard_gate(self, qubit_index)`
- **Purpose**: Applies a Hadamard gate to a specified qubit.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit.
- **Usage**: Used to create a superposition state, allowing the qubit to be both 0 and 1 simultaneously.

## `apply_cnot_gate(self, control_qubit_index, target_qubit_index)`
- **Purpose**: Applies a Controlled-NOT (CNOT) gate between two qubits.
- **Parameters**:
    - `control_qubit_index` (int): Index of the control qubit.
    - `target_qubit_index` (int): Index of the target qubit.
- **Usage**: Fundamental for entangling qubits, which is essential for quantum algorithms.

## `apply_toffoli_gate(self, control_qubit_index1, control_qubit_index2, target_qubit_index)`
- **Purpose**: Applies a Toffoli gate (CCX gate) to three qubits.
- **Parameters**:
    - `control_qubit_index1` (int): Index of the first control qubit.
    - `control_qubit_index2` (int): Index of the second control qubit.
    - `target_qubit_index` (int): Index of the target qubit.
- **Usage**: Acts as a quantum AND gate, used in algorithms requiring conditional logic.

## `apply_swap_gate(self, qubit_index1, qubit_index2)`
- **Purpose**: Swaps the states of two qubits.
- **Parameters**:
    - `qubit_index1` (int): Index of the first qubit.
    - `qubit_index2` (int): Index of the second qubit.
- **Usage**: Useful in quantum algorithms for rearranging qubit states.

## `apply_cswap_gate(self, control_qubit_index, target_qubit_index1, target_qubit_index2)`
- **Purpose**: Applies a controlled-SWAP (Fredkin) gate that swaps two targets when the control is |1⟩.
- **Parameters**:
    - `control_qubit_index` (int): Index of the control qubit.
    - `target_qubit_index1` (int): Index of the first target qubit.
    - `target_qubit_index2` (int): Index of the second target qubit.
- **Usage**: Used in overlap estimation routines such as the swap test.

## `apply_pauli_x_gate(self, qubit_index)`
- **Purpose**: Applies a Pauli-X gate to a specified qubit.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit.
- **Usage**: Equivalent to a NOT gate, flips the qubit state.

## `apply_pauli_y_gate(self, qubit_index)`
- **Purpose**: Applies a Pauli-Y gate to a specified qubit.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit.
- **Usage**: Impacts the phase and amplitude of a qubit’s state.

## `apply_pauli_z_gate(self, qubit_index)`
- **Purpose**: Applies a Pauli-Z gate to a specified qubit.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit.
- **Usage**: Alters the phase of a qubit without changing its amplitude.

## `apply_t_gate(self, qubit_index)`
- **Purpose**: Applies the T (π/8) phase gate to a specified qubit.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit.
- **Usage**: Adds a π/4 phase to |1⟩. Together with the Hadamard (H) and CNOT gates, it enables universal single-qubit control.

## `execute_circuit(self)`
- **Purpose**: Executes the quantum circuit and retrieves the results.
- **Usage**: Used to run the entire set of quantum operations and measure the outcomes.

## `get_final_state_vector(self)`
- **Purpose**: Returns the final state vector of the circuit from the configured backend.
- **Usage**: Retrieves the full quantum state for simulation and analysis workflows.

## `draw_circuit(self)`
- **Purpose**: Visualizes the quantum circuit.
- **Returns**: A string representation of the circuit visualization (format depends on backend).
- **Usage**: Returns a visualization string that can be printed or used programmatically. Example: `print(qc.draw_circuit())` or `viz = qc.draw_circuit()`.
- **Note**: Uses underlying libraries' methods for drawing circuits (Qiskit's `draw()`, Cirq's `str()`, or Braket's `str()`).

## `apply_rx_gate(self, qubit_index, angle)`
- **Purpose**: Applies a rotation around the X-axis to a specified qubit with an optional parameter for optimization.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit.
    - `angle` (str or float): Angle in radians for the rotation. Can be a static value or a parameter name for optimization.
- **Usage**: Used to rotate a qubit around the X-axis, often in parameterized quantum circuits for variational algorithms.

## `apply_ry_gate(self, qubit_index, angle)`
- **Purpose**: Applies a rotation around the Y-axis to a specified qubit with an optional parameter for optimization.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit.
    - `angle` (str or float): Angle in radians for the rotation. Can be a static value or a parameter name for optimization.
- **Usage**: Used to rotate a qubit around the Y-axis in parameterized circuits, aiding in the creation of complex quantum states.

## `apply_rz_gate(self, qubit_index, angle)`
- **Purpose**: Applies a rotation around the Z-axis to a specified qubit with an optional parameter for optimization.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit.
    - `angle` (str or float): Angle in radians for the rotation. Can be a static value or a parameter name for optimization.
- **Usage**: Utilized in parameterized quantum circuits to modify the phase of a qubit state during optimization.

## `apply_u_gate(self, qubit_index, theta, phi, lambd)`
- **Purpose**: Applies the universal single-qubit U(θ, φ, λ) gate.
- **Parameters**:
    - `qubit_index` (int): Index of the qubit.
    - `theta` (float): Rotation angle θ.
    - `phi` (float): Rotation angle φ.
    - `lambd` (float): Rotation angle λ.
- **Usage**: Provides full single-qubit unitary control via Z–Y–Z Euler decomposition.

## `execute_circuit(self, parameter_values=None)`
- **Purpose**: Executes the quantum circuit with the ability to bind specific parameter values if provided.
- **Parameters**:
    - `parameter_values` (dict, optional): A dictionary where keys are parameter names and values are the numerical values to bind.
- **Usage**: Enables the execution of parameterized circuits by binding parameter values, facilitating optimization processes.

## `bind_parameters(self, parameter_values)`
- **Purpose**: Binds numerical values to the parameters of the quantum circuit, allowing for dynamic updates during optimization.
- **Parameters**:
    - `parameter_values` (dict): A dictionary with parameter names as keys and numerical values to bind.
- **Usage**: Essential for optimization loops where parameters are adjusted based on cost function evaluations.

## `_handle_parameter(self, param_name)`
- **Purpose**: Internal function to manage parameter registration.
- **Parameters**:
    - `param_name` (str): The name of the parameter to handle.
- **Usage**: Automatically invoked when applying parameterized gates to keep track of parameters efficiently.

## `swap_test(self, ancilla_qubit, qubit1, qubit2)`
- **Purpose**: Builds the swap-test subcircuit (H–CSWAP–H) to compare two quantum states.
- **Parameters**:
    - `ancilla_qubit` (int): Index of the ancilla control qubit.
    - `qubit1` (int): Index of the first state qubit.
    - `qubit2` (int): Index of the second state qubit.
- **Usage**: Used in overlap/fidelity estimation between two states.

## `measure_overlap(self, qubit1, qubit2, ancilla_qubit=0)`
- **Purpose**: Executes the swap test and returns |⟨ψ|φ⟩|² using backend-specific measurement parsing.
- **Parameters**:
    - `qubit1` (int): Index of the first state qubit.
    - `qubit2` (int): Index of the second state qubit.
    - `ancilla_qubit` (int, default to 0): Index of the ancilla qubit.
- **Usage**: Convenience wrapper for fidelity/overlap measurement across backends.

## NumPy Usage in QDP

QDP internally uses NumPy arrays (`numpy.ndarray`) to represent quantum state vectors,
especially within simulation backends and testing utilities.

### State Vector Representation

A quantum state for `n` qubits is represented as a 1-dimensional NumPy array:

- Length: `2**n`
- Data type: complex numbers (commonly `complex128`)

### Example

Below is an example of a 3-qubit computational basis state |000⟩ represented using NumPy:

```python
import numpy as np

state_vector = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0],
    dtype=np.complex128
)
```
### Notes

- NumPy-based state vectors are primarily used in simulation backends.
- Test utilities within QDP frequently rely on NumPy for validation.