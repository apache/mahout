<a id="qumat"></a>

# qumat

<a id="qumat.QuMat"></a>

## QuMat Objects

```python
class QuMat()
```

Unified interface for quantum circuit operations across multiple backends.

Provides a consistent API for creating and manipulating quantum circuits
using different quantum computing backends (Qiskit, Cirq, Amazon Braket).
Abstracts backend-specific details for gate operations, circuit execution,
and state measurement.

**Arguments**:

- `backend_config` (`dict`): Configuration dictionary for the quantum backend.
Must contain ``backend_name`` (str) and ``backend_options`` (dict).
The ``backend_options`` should include ``simulator_type`` and ``shots``.

<a id="qumat.QuMat.__init__"></a>

#### \_\_init\_\_

```python
def __init__(backend_config)
```

Create a QuMat instance with the specified backend configuration.

**Arguments**:

- `backend_config` (`dict`): Configuration dictionary containing backend name
and options. Required keys:
- ``backend_name``: Name of the backend (e.g., "qiskit", "cirq", "amazon_braket")
- ``backend_options``: Dictionary with backend-specific options

**Raises**:

- `ImportError`: If the specified backend module cannot be imported.
- `ValueError`: If backend_config is not a dictionary.
- `KeyError`: If required configuration keys are missing.

<a id="qumat.QuMat.create_empty_circuit"></a>

#### create\_empty\_circuit

```python
def create_empty_circuit(num_qubits: int | None = None)
```

Create an empty quantum circuit with the specified number of qubits.

Must be called before applying any gates or executing operations.

**Arguments**:

- `num_qubits` (`int | None, optional`): Number of qubits in the circuit. If ``None``,
creates a circuit without pre-allocated qubits.

<a id="qumat.QuMat.apply_not_gate"></a>

#### apply\_not\_gate

```python
def apply_not_gate(qubit_index)
```

Apply a NOT gate (Pauli-X gate) to the specified qubit.

Flips the qubit state from |0⟩ to |1⟩ or |1⟩ to |0⟩.
Equivalent to the Pauli-X gate.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_hadamard_gate"></a>

#### apply\_hadamard\_gate

```python
def apply_hadamard_gate(qubit_index)
```

Apply a Hadamard gate to the specified qubit.

Creates a superposition state, transforming |0⟩ to (|0⟩ + |1⟩)/√2
and |1⟩ to (|0⟩ - |1⟩)/√2.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_cnot_gate"></a>

#### apply\_cnot\_gate

```python
def apply_cnot_gate(control_qubit_index, target_qubit_index)
```

Apply a Controlled-NOT (CNOT) gate between two qubits.

Fundamental for entangling qubits. Flips the target qubit if and only
if the control qubit is in the |1⟩ state.

**Arguments**:

- `control_qubit_index` (`int`): Index of the control qubit.
- `target_qubit_index` (`int`): Index of the target qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_toffoli_gate"></a>

#### apply\_toffoli\_gate

```python
def apply_toffoli_gate(control_qubit_index1, control_qubit_index2,
                       target_qubit_index)
```

Apply a Toffoli gate (CCX gate) to three qubits.

Acts as a quantum AND gate. Flips the target qubit if and only if
both control qubits are in the |1⟩ state.

**Arguments**:

- `control_qubit_index1` (`int`): Index of the first control qubit.
- `control_qubit_index2` (`int`): Index of the second control qubit.
- `target_qubit_index` (`int`): Index of the target qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_swap_gate"></a>

#### apply\_swap\_gate

```python
def apply_swap_gate(qubit_index1, qubit_index2)
```

Swap the states of two qubits.

**Arguments**:

- `qubit_index1` (`int`): Index of the first qubit.
- `qubit_index2` (`int`): Index of the second qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_cswap_gate"></a>

#### apply\_cswap\_gate

```python
def apply_cswap_gate(control_qubit_index, target_qubit_index1,
                     target_qubit_index2)
```

Apply a controlled-SWAP (Fredkin) gate.

Swaps the states of two target qubits if and only if the control
qubit is in the |1⟩ state.

**Arguments**:

- `control_qubit_index` (`int`): Index of the control qubit.
- `target_qubit_index1` (`int`): Index of the first target qubit.
- `target_qubit_index2` (`int`): Index of the second target qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_pauli_x_gate"></a>

#### apply\_pauli\_x\_gate

```python
def apply_pauli_x_gate(qubit_index)
```

Apply a Pauli-X gate to the specified qubit.

Equivalent to the NOT gate. Flips the qubit state from |0⟩ to |1⟩
or |1⟩ to |0⟩.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_pauli_y_gate"></a>

#### apply\_pauli\_y\_gate

```python
def apply_pauli_y_gate(qubit_index)
```

Apply a Pauli-Y gate to the specified qubit.

Rotates the qubit around the Y-axis of the Bloch sphere, affecting
both phase and amplitude.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_pauli_z_gate"></a>

#### apply\_pauli\_z\_gate

```python
def apply_pauli_z_gate(qubit_index)
```

Apply a Pauli-Z gate to the specified qubit.

Rotates the qubit around the Z-axis of the Bloch sphere, altering
the phase without changing the amplitude.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_t_gate"></a>

#### apply\_t\_gate

```python
def apply_t_gate(qubit_index)
```

Apply a T-gate (π/8 gate) to the specified qubit.

Applies a relative pi/4 phase (multiplies the |1> state by e^{i*pi/4}).
Essential for universal quantum computation when combined with
Hadamard and CNOT gates.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.execute_circuit"></a>

#### execute\_circuit

```python
def execute_circuit(parameter_values=None)
```

Execute the quantum circuit and return the measurement results.

Runs the circuit on the configured backend. For parameterized circuits,
provide parameter values to bind before execution.

**Arguments**:

- `parameter_values` (`dict, optional`): Dictionary mapping parameter names to numerical
values. Binds these values to circuit parameters before execution.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

**Returns**:

`dict | list[dict]`: Measurement results. Format depends on the backend:
- Qiskit/Braket: Dictionary with state strings as keys and counts as values
- Cirq: List of dictionaries with integer states as keys

<a id="qumat.QuMat.bind_parameters"></a>

#### bind\_parameters

```python
def bind_parameters(parameter_values)
```

Bind numerical values to circuit parameters.

Assigns numerical values to symbolic parameters defined in parameterized
gates.

**Arguments**:

- `parameter_values` (`dict`): Dictionary mapping parameter names to numerical
values.

**Raises**:

- `ValueError`: If a parameter name is not found in the circuit's
parameter list.

<a id="qumat.QuMat.get_final_state_vector"></a>

#### get\_final\_state\_vector

```python
def get_final_state_vector()
```

Return the final state vector of the quantum circuit.

The complete quantum state vector after circuit execution,
representing the full quantum state of all qubits. For parameterized
circuits, call bind_parameters() first to set parameter values.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.
- `ValueError`: If parameterized circuit has unbound parameters.

**Returns**:

`numpy.ndarray`: The final state vector as a numpy array.

<a id="qumat.QuMat.draw_circuit"></a>

#### draw\_circuit

```python
def draw_circuit()
```

Visualize the quantum circuit.

Generates a visual representation of the circuit. The output format
depends on the backend implementation.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

**Returns**:

`str | object`: Circuit visualization. The exact type depends on the backend.

<a id="qumat.QuMat.draw"></a>

#### draw

```python
def draw()
```

Alias for draw_circuit() for convenience.

Provides a shorter method name that matches common quantum computing
library conventions and documentation examples.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

**Returns**:

`str | object`: Circuit visualization. The exact type depends on the backend.

<a id="qumat.QuMat.apply_rx_gate"></a>

#### apply\_rx\_gate

```python
def apply_rx_gate(qubit_index, angle)
```

Apply a rotation around the X-axis to the specified qubit.

Rotates the qubit by the given angle around the X-axis of the Bloch
sphere. The angle can be a static value or a parameter name for
parameterized circuits.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.
- `angle` (`float | str`): Rotation angle in radians. Can be a float or a string
parameter name.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_ry_gate"></a>

#### apply\_ry\_gate

```python
def apply_ry_gate(qubit_index, angle)
```

Apply a rotation around the Y-axis to the specified qubit.

Rotates the qubit by the given angle around the Y-axis of the Bloch
sphere. The angle can be a static value or a parameter name for
parameterized circuits.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.
- `angle` (`float | str`): Rotation angle in radians. Can be a float or a string
parameter name.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_rz_gate"></a>

#### apply\_rz\_gate

```python
def apply_rz_gate(qubit_index, angle)
```

Apply a rotation around the Z-axis to the specified qubit.

Rotates the qubit by the given angle around the Z-axis of the Bloch
sphere. The angle can be a static value or a parameter name for
parameterized circuits.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.
- `angle` (`float | str`): Rotation angle in radians. Can be a float or a string
parameter name.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.apply_u_gate"></a>

#### apply\_u\_gate

```python
def apply_u_gate(qubit_index, theta, phi, lambd)
```

Apply a U gate (universal single-qubit gate) to the specified qubit.

A universal single-qubit gate parameterized by three angles (theta,
phi, lambd) that can represent any single-qubit unitary operation.

**Arguments**:

- `qubit_index` (`int`): Index of the qubit.
- `theta` (`float`): First rotation angle in radians.
- `phi` (`float`): Second rotation angle in radians.
- `lambd` (`float`): Third rotation angle in radians.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.swap_test"></a>

#### swap\_test

```python
def swap_test(ancilla_qubit, qubit1, qubit2)
```

Implement the swap test circuit for measuring overlap between two quantum states.

Measures the inner product between the states on ``qubit1`` and ``qubit2``.
The probability of measuring the ancilla qubit in state |0⟩ is related
to the overlap as: P(0) = (1 + |⟨ψ|φ⟩|²) / 2

**Arguments**:

- `ancilla_qubit` (`int`): Index of the ancilla qubit (should be initialized to |0⟩).
- `qubit1` (`int`): Index of the first qubit containing state |ψ⟩.
- `qubit2` (`int`): Index of the second qubit containing state |φ⟩.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

<a id="qumat.QuMat.measure_overlap"></a>

#### measure\_overlap

```python
def measure_overlap(qubit1, qubit2, ancilla_qubit=0)
```

Measure the overlap (fidelity) between two quantum states using the swap test.

Creates a swap test circuit to calculate the similarity between the
quantum states on ``qubit1`` and ``qubit2``. Returns the squared overlap
|⟨ψ|φ⟩|², which represents the fidelity between the two states.

The swap test measures P(ancilla=0), related to overlap as:
P(0) = (1 + |⟨ψ|φ⟩|²) / 2

For certain states (especially identical excited states), global phase
effects may cause the ancilla to measure predominantly |1⟩ instead of |0⟩.
This method handles both cases by taking the measurement probability
closer to 1.

**Arguments**:

- `qubit1` (`int`): Index of the first qubit containing state |ψ⟩.
- `qubit2` (`int`): Index of the second qubit containing state |φ⟩.
- `ancilla_qubit` (`int, optional`): Index of the ancilla qubit. Default is 0. Should be
initialized to |0⟩.

**Raises**:

- `RuntimeError`: If the circuit has not been initialized.

**Returns**:

`float`: The squared overlap |⟨ψ|φ⟩|² between the two states (fidelity),
clamped to the range [0.0, 1.0].

<a id="qumat.QuMat.calculate_prob_zero"></a>

#### calculate\_prob\_zero

```python
def calculate_prob_zero(results, ancilla_qubit)
```

Calculate the probability of measuring the ancilla qubit in |0⟩ state.

Delegates to the backend-specific implementation. Different backends
may use different qubit ordering conventions (little-endian vs big-endian).

**Arguments**:

- `results` (`dict | list[dict]`): Measurement results from ``execute_circuit()``. Format
depends on the backend.
- `ancilla_qubit` (`int`): Index of the ancilla qubit.

**Returns**:

`float`: Probability of measuring the ancilla qubit in |0⟩ state.

