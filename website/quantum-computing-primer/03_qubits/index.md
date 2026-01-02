---
layout: page
title: Quantum Bits (Qubits)
---

# 3. Quantum Bits (Qubits)

## 3.1 Classical Bits vs. Qubits

In classical computing, the fundamental unit of information is the **bit**, which can exist in one of two states: `0` or `1`. Quantum computing, however, introduces the concept of a **qubit**, which can exist in a **superposition** of both states simultaneously. This means a qubit can be in a state that is a combination of `0` and `1`, represented as:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

{% raw %}
where $\alpha$ and $\beta$ are complex numbers representing the probability
amplitudes of the qubit being in the $|0\rangle$ and $|1\rangle$ states,
respectively. The probabilities of measuring the qubit in either state are given
by $|\alpha|^2$ and $|\beta|^2$, and they must satisfy the normalization condition:
{% endraw %}

$$|\alpha|^2 + |\beta|^2 = 1$$

## 3.2 Representing Qubits

Qubits can be visualized using the **Bloch sphere**, a geometric representation
of the quantum state of a single qubit. The Bloch sphere is a unit sphere where
the north and south poles represent the $|0\rangle$ and $|1\rangle$ states,
respectively. Any point on the surface of the sphere represents a valid quantum
state of the qubit.

The state of a qubit can also be described using a **state vector** in a
two-dimensional complex vector space. For example, the state $|0\rangle$ is
represented as:

$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

{% raw %}
and the state $|1\rangle$ is represented as:
{% endraw %}

$$|1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

## 3.3 Creating Qubits with `qumat`

In `qumat`, qubits are created by initializing a quantum circuit with a specified number of qubits. The `create_empty_circuit` function is used to create a circuit with a given number of qubits. Here's an example of creating a quantum circuit with a single qubit:

```python
from qumat import QuMat

# Initialize the quantum circuit with a single qubit
backend_config = {
    'backend_name': 'qiskit',  # Choose the backend (e.g., 'qiskit', 'cirq', 'amazon_braket')
    'backend_options': {
        'simulator_type': 'qasm_simulator',  # Type of simulator
        'shots': 1000  # Number of shots (measurements)
    }
}

qc = QuMat(backend_config)
qc.create_empty_circuit(1)  # Create a circuit with 1 qubit
```

In this example, we initialize a quantum circuit with one qubit using the qiskit backend. The create_empty_circuit function sets up the circuit, and we can now apply quantum gates to manipulate the qubit.

### Example: Applying a Hadamard Gate

The Hadamard gate ((H)) is a fundamental quantum gate that puts a qubit into a
superposition state. Applying the Hadamard gate to a qubit initially in the
$|0\rangle$ state results in the state:

$$H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

Here's how you can apply a Hadamard gate to a qubit using qumat:

```python
# Apply the Hadamard gate to the first qubit (index 0)
qc.apply_hadamard_gate(0)

# Execute the circuit and get the measurement results
result = qc.execute_circuit()
print(result)
```

In this example, the Hadamard gate is applied to the qubit at index 0, and the
circuit is executed to obtain the measurement results. The output will show the
probabilities of measuring the qubit in the $|0\rangle$ and $|1\rangle$ states.

### Visualizing the Circuit

You can also visualize the quantum circuit using the draw method:

```python
# Draw the circuit
qc.draw()
```

This returns a textual representation of the circuit, which you can print with `print(qc.draw())` or use programmatically. The visualization shows the sequence of gates applied to the qubits.

---

This section introduced the concept of qubits, their representation, and how to create and manipulate them using the qumat library. In the next section, we will explore quantum gates in more detail and learn how to apply them to perform quantum operations.
