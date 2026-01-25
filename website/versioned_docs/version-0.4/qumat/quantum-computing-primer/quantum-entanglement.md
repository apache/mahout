---
title: Quantum Entanglement
---

# 6. Quantum Entanglement

## 6.1 Understanding Entanglement

Quantum entanglement is one of the most fascinating and counterintuitive phenomena in quantum mechanics. When two or more qubits become entangled, the state of one qubit becomes directly related to the state of the other, no matter how far apart they are. This means that measuring one qubit instantly determines the state of the other, even if they are light-years apart.

### Key Concepts:
- **Entangled States**: A quantum state of two or more qubits that cannot be described independently of each other.
- **Bell States**: Specific examples of maximally entangled quantum states of two qubits.
- **Non-Locality**: The idea that entangled particles can influence each other instantaneously, regardless of distance.

## 6.2 Entanglement with `qumat`

In this section, we will explore how to create and measure entangled states using the `qumat` library. We will start by creating a simple entangled state known as a Bell state, which is a maximally entangled quantum state of two qubits.

### Example: Creating a Bell State

A Bell state can be created by applying a Hadamard gate to the first qubit, followed by a CNOT gate with the first qubit as the control and the second qubit as the target. This results in a state where the two qubits are perfectly correlated.

```python
from qumat import QuMat

# Initialize the quantum circuit with 2 qubits
backend_config = {'backend_name': 'qiskit', 'backend_options': {'simulator_type': 'qasm_simulator', 'shots': 1000}}
qc = QuMat(backend_config)
qc.create_empty_circuit(2)

# Apply a Hadamard gate to the first qubit
qc.apply_hadamard_gate(0)

# Apply a CNOT gate with the first qubit as control and the second as target
qc.apply_cnot_gate(0, 1)

# Execute the circuit and measure the results
result = qc.execute_circuit()
print(result)
```

### Expected Output:
The output will show the measurement results of the two qubits. Since the qubits are entangled, you should observe that the states of the two qubits are perfectly correlated. For example, if the first qubit is measured as `0`, the second qubit will also be `0`, and if the first qubit is `1`, the second qubit will also be `1`.

### Visualizing the Circuit:
You can also visualize the circuit to better understand the sequence of operations:

```python
qc.draw()
```

### Explanation:
- **Hadamard Gate**: Creates a superposition of the first qubit.
- **CNOT Gate**: Entangles the two qubits, creating a Bell state.

### 6.2.1 Measuring Entangled Qubits

Once the qubits are entangled, measuring one qubit will instantly determine the state of the other. This is a key feature of quantum entanglement and is used in various quantum algorithms and protocols.

### Example: Measuring Entangled Qubits

```python
# Execute the circuit and measure the results
result = qc.execute_circuit()
print(result)
```

### Expected Output:
The output will show the measurement counts for the two qubits. Since the qubits are entangled, the results will show a strong correlation between the states of the two qubits.

### 6.2.2 Applications of Entanglement

Quantum entanglement is a fundamental resource in quantum computing and is used in various applications, including:
- **Quantum Teleportation**: Transmitting quantum information from one location to another.
- **Quantum Cryptography**: Securely sharing encryption keys using entangled qubits.
- **Quantum Error Correction**: Protecting quantum information from errors using entangled states.

### Example: Quantum Teleportation

Quantum teleportation is a protocol that allows the transfer of quantum information from one qubit to another, even if they are far apart. This is achieved using entanglement and classical communication.

```python
# Example implementation of quantum teleportation
qc.create_empty_circuit(3)

# Create an entangled pair between qubit 1 and qubit 2
qc.apply_hadamard_gate(1)
qc.apply_cnot_gate(1, 2)

# Prepare the qubit to be teleported (qubit 0)
qc.apply_hadamard_gate(0)

# Perform the teleportation protocol
qc.apply_cnot_gate(0, 1)
qc.apply_hadamard_gate(0)
qc.apply_cnot_gate(1, 2)
qc.apply_toffoli_gate(0, 1, 2)

# Measure the qubits
result = qc.execute_circuit()
print(result)
```

### Expected Output:
The output will show the measurement results, demonstrating that the state of qubit 0 has been successfully teleported to qubit 2.

## 6.3 Conclusion

Quantum entanglement is a powerful and essential concept in quantum computing. By understanding how to create and manipulate entangled states using `qumat`, you can begin to explore more advanced quantum algorithms and applications. In the next section, we will delve into quantum algorithms, starting with the Deutsch-Jozsa algorithm.
