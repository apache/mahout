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

## `execute_circuit(self)`
- **Purpose**: Executes the quantum circuit and retrieves the results.
- **Usage**: Used to run the entire set of quantum operations and measure the outcomes.

## `draw_circuit(self)`
- **Purpose**: Visualizes the quantum circuit.
- **Usage**: Provides a graphical representation of the quantum circuit for better understanding.
- **Note**: Just a pass through function, will use underlying libraries 
  method for drawing circuit. 

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
  
