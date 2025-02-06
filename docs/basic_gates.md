# Quantum Gate Explanations

## NOT Gate (X-Gate)
The NOT gate, also called the **Pauli X Gate**, is a fundamental quantum gate used to flip the state of a qubit. It is often referred to as a bit-flip gate. When applied to a qubit in the |0⟩ state, it changes it to the |1⟩ state, and vice versa. Mathematically, if |ψ⟩ represents the qubit's state, applying the X-gate results in:

\[ X|ψ⟩ = |¬ψ⟩ \]

## Hadamard Gate (H-Gate)
The Hadamard gate, denoted as the H-gate, is used to create superposition states. When applied to a qubit in the |0⟩ state, it transforms it into an equal superposition of |0⟩ and |1⟩ states. Mathematically:


H|0⟩ = (|0⟩ + |1⟩) / √2



This gate is crucial in quantum algorithms like Grover's and quantum teleportation.

## CNOT Gate (Controlled-X Gate)
The CNOT gate, or controlled-X gate, is an entangling gate that acts on two qubits: a control qubit and a target qubit. If the control qubit is in the |1⟩ state, it applies the X-gate to the target qubit; otherwise, it does nothing. It creates entanglement between the two qubits and is essential in building quantum circuits for various applications.

## Toffoli Gate (Controlled-Controlled-X Gate)
The Toffoli gate is a three-qubit gate that acts as a controlled-controlled-X gate. It applies the X-gate to the target qubit only if both control qubits are in the |1⟩ state. It is used in quantum computing for implementing reversible classical logic operations.

## SWAP Gate
The SWAP gate exchanges the states of two qubits. When applied, it swaps the state of the first qubit with the state of the second qubit. This gate is fundamental in quantum algorithms and quantum error correction codes.

## Pauli Y Gate (Y-Gate)
The Pauli Y gate introduces complex phase shifts along with a bit-flip operation. It can be thought of as a combination of bit-flip and phase-flip gates. Mathematically:

\[ Y|0⟩ = i|1⟩ \]
\[ Y|1⟩ = -i|0⟩ \]

It's essential in quantum error correction and quantum algorithms.

## Pauli Z Gate (Z-Gate)
The Pauli Z gate introduces a phase flip without changing the qubit's state. It leaves |0⟩ unchanged and transforms |1⟩ to -|1⟩. Mathematically:

\[ Z|0⟩ = |0⟩ \]
\[ Z|1⟩ = -|1⟩ \]

It's used for measuring the phase of a qubit.

## T-Gate (π/8 Gate) (New Addition)
The T-Gate applies a **π/4 phase shift** to the qubit. It is essential for quantum computing because it, along with the Hadamard and CNOT gates, allows for **universal quantum computation**. Mathematically:

\[ T|0⟩ = |0⟩ \]
\[ T|1⟩ = e^{i\pi/4} |1⟩ \]

# **Updates**
- **Acknowledged support for Cirq & Braket** (New Addition)
- **Removed Pauli X Gate** (Merged into NOT Gate)
- **Added T-Gate** (New Addition)
- **Fixed typos**

These quantum gates are fundamental building blocks in quantum computing, enabling the manipulation and transformation of qubit states to perform various quantum algorithms and computations.
