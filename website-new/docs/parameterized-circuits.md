---
title: Parameterized Circuits
---
# Parameterized Quantum Circuits and Rotation Gates

Parameterized quantum circuits (PQCs) contain gates with tunable parameters that can be optimized. Instead of using fixed angles, you use symbolic parameters that can be adjusted to find optimal values.

**Why use parameters?** Fixed angles give you one specific operation. Parameters let you:
- **Optimize**: Find the best angles for your task
- **Train**: Adjust parameters based on results (like machine learning)
- **Explore**: Try different values without rewriting the circuit

Rotation gates (Rx, Ry, Rz) are the primary parameterized gates. Use string names (e.g., `"theta"`) instead of numbers to create parameters, then bind values before execution.

## Rotation Gates

Rotation gates rotate a qubit around the X, Y, or Z axis of the Bloch sphere. Each axis controls different aspects:
- **X-axis (Rx)**: Bit-flip rotations
- **Y-axis (Ry)**: Creates superpositions (like Hadamard)
- **Z-axis (Rz)**: Phase rotations (doesn't change probabilities)

### Rx Gate

Rotates a qubit around the X-axis by angle $\theta$.

$$R_x(\theta) = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

$$R_x(\pi/2)|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle)$$

```python
from qumat import QuMat
import numpy as np

backend_config = {
    'backend_name': 'qiskit',
    'backend_options': {'simulator_type': 'qasm_simulator', 'shots': 1000}
}
qumat = QuMat(backend_config)
qumat.create_empty_circuit(num_qubits=1)
qumat.apply_rx_gate(0, angle=np.pi / 2)
results = qumat.execute_circuit()
```

### Ry Gate

Rotates a qubit around the Y-axis by angle $\theta$.

$$R_y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

$$R_y(\pi/2)|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

```python
qumat.create_empty_circuit(num_qubits=1)
qumat.apply_ry_gate(0, angle=np.pi / 2)  # Creates (|0⟩ + |1⟩)/√2
results = qumat.execute_circuit()
```

### Rz Gate

Rotates a qubit around the Z-axis by angle $\theta$. Changes phase without affecting probability amplitudes.

$$R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

$$R_z(\pi/2)\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$$

Probabilities remain: P(|0⟩) = P(|1⟩) = 1/2

```python
qumat.create_empty_circuit(num_qubits=1)
qumat.apply_hadamard_gate(0)
qumat.apply_rz_gate(0, angle=np.pi / 2)
results = qumat.execute_circuit()
```

**Notes**: Angles are in **radians** (π = 180°, π/2 = 90°, π/4 = 45°). Supported across all backends (Qiskit, Cirq, Braket).

## Creating Parameterized Circuits

**Fixed value** (angle is set once):
```python
qumat.apply_ry_gate(0, angle=0.5)  # Fixed angle
```

**Parameterized** (angle can be changed):
```python
qumat.apply_ry_gate(0, angle="theta")  # Parameter - can be optimized
```

Use string names instead of numbers to create parameters:

```python
from qumat import QuMat
import numpy as np

backend_config = {
    'backend_name': 'qiskit',
    'backend_options': {'simulator_type': 'qasm_simulator', 'shots': 1000}
}
qumat = QuMat(backend_config)

# Step 1: Create circuit with parameterized gates
qumat.create_empty_circuit(num_qubits=2)
qumat.apply_ry_gate(0, angle="theta")  # String = parameter
qumat.apply_rz_gate(0, angle="phi")
qumat.apply_rx_gate(1, angle="alpha")

# Step 2: Parameters are automatically registered
print(list(qumat.parameters.keys()))  # ['theta', 'phi', 'alpha']

# Step 3: Bind parameters before execution
qumat.bind_parameters({
    "theta": np.pi / 4,
    "phi": np.pi / 2,
    "alpha": np.pi / 3
})

# Step 4: Execute with bound parameters
results = qumat.execute_circuit()

# Alternative: Bind during execution (skips step 3)
results = qumat.execute_circuit(parameter_values={
    "theta": np.pi / 4,
    "phi": np.pi / 2,
    "alpha": np.pi / 3
})
```

## bind_parameters()

Assigns numerical values to symbolic parameters. Must be called before `execute_circuit()` if parameters are used.

**Raises**: `ValueError` if parameter name not found.

```python
# Create circuit with parameters
qumat.create_empty_circuit(num_qubits=1)
qumat.apply_ry_gate(0, angle="theta")
qumat.apply_rz_gate(0, angle="phi")

# Bind values to parameters
qumat.bind_parameters({"theta": 0.5, "phi": 1.0})

# Now you can execute
results = qumat.execute_circuit()
```

## Optimization Loops

The power of parameterized circuits: optimize parameters to minimize a cost function. The workflow is:
1. Create circuit with parameters
2. Try different parameter values
3. Execute and compute cost
4. Find values that minimize cost

Example:

```python
from qumat import QuMat
import numpy as np

backend_config = {
    'backend_name': 'qiskit',
    'backend_options': {'simulator_type': 'qasm_simulator', 'shots': 1000}
}
qumat = QuMat(backend_config)

# Create parameterized circuit once
qumat.create_empty_circuit(num_qubits=2)
qumat.apply_ry_gate(0, angle="theta")
qumat.apply_cnot_gate(0, 1)
qumat.apply_rz_gate(1, angle="phi")

# Simple optimization loop
best_cost = float('inf')
best_params = None

for iteration in range(10):
    # Try different parameter values
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    # Bind and execute
    qumat.bind_parameters({"theta": theta, "phi": phi})
    results = qumat.execute_circuit()

    # Compute cost (example: minimize probability of |00⟩)
    total_shots = sum(results.values())
    prob_00 = results.get("00", 0) / total_shots
    cost = prob_00

    # Track best result
    if cost < best_cost:
        best_cost = cost
        best_params = {"theta": theta, "phi": phi}

print(f"Best parameters: {best_params}, cost: {best_cost}")

# With SciPy optimizer
from scipy.optimize import minimize

def cost_function(params):
    theta, phi = params
    qumat.bind_parameters({"theta": theta, "phi": phi})
    results = qumat.execute_circuit()
    total_shots = sum(results.values())
    prob_00 = results.get("00", 0) / total_shots
    return prob_00

result = minimize(cost_function, [0.5, 0.5], method='COBYLA')
print(f"Optimized: {result.x}, cost: {result.fun}")
```

## Key Points

- **Parameter creation**: Use string names (e.g., `"theta"`) in rotation gates to create parameters
- **Auto-registration**: Parameters are automatically registered when first used
- **Binding required**: All parameters must be bound (assigned values) before execution
- **Value types**: Parameter values must be numerical (float or int)
- **Backend support**: Works with all backends (Qiskit, Cirq, Braket)
- **Reuse**: Create circuit once, bind different parameter values for optimization loops
