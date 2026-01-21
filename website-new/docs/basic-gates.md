---
title: Basic Gates
---
# Quantum Gate Explanations

## NOT Gate (X-Gate)
The NOT gate, also called the **Pauli X Gate**, is a fundamental quantum gate used to flip the state of a qubit. It is often referred to as a bit-flip gate. When applied to a qubit in the $|0\rangle$ state, it changes it to the $|1\rangle$ state, and vice versa. Mathematically, if $|\psi\rangle$ represents the qubit's state, applying the X-gate results in:

$$X|\psi\rangle = |\neg\psi\rangle$$

## Hadamard Gate (H-Gate)
The Hadamard gate, denoted as the H-gate, is used to create superposition states. When applied to a qubit in the $|0\rangle$ state, it transforms it into an equal superposition of $|0\rangle$ and $|1\rangle$ states. Mathematically:

$$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$



This gate is crucial in quantum algorithms like Grover's and quantum teleportation.

## CNOT Gate (Controlled-X Gate)
The CNOT gate, or controlled-X gate, is an entangling gate that acts on two qubits: a control qubit and a target qubit. If the control qubit is in the $|1\rangle$ state, it applies the X-gate to the target qubit; otherwise, it does nothing. It creates entanglement between the two qubits and is essential in building quantum circuits for various applications.

## Toffoli Gate (Controlled-Controlled-X Gate)
The Toffoli gate is a three-qubit gate that acts as a controlled-controlled-X gate. It applies the X-gate to the target qubit only if both control qubits are in the $|1\rangle$ state. It is used in quantum computing for implementing reversible classical logic operations.

## SWAP Gate
The SWAP gate exchanges the states of two qubits. When applied, it swaps the state of the first qubit with the state of the second qubit. This gate is fundamental in quantum algorithms and quantum error correction codes.

## Pauli Y Gate (Y-Gate)
The Pauli Y gate introduces complex phase shifts along with a bit-flip operation. It can be thought of as a combination of bit-flip and phase-flip gates. Mathematically:

$$Y|0\rangle = i|1\rangle$$
$$Y|1\rangle = -i|0\rangle$$

It's essential in quantum error correction and quantum algorithms.

## Pauli Z Gate (Z-Gate)
The Pauli Z gate introduces a phase flip without changing the qubit's state. It leaves $|0\rangle$ unchanged and transforms $|1\rangle$ to $-|1\rangle$. Mathematically:

$$Z|0\rangle = |0\rangle$$
$$Z|1\rangle = -|1\rangle$$

It's used for measuring the phase of a qubit.

## T-Gate (π/8 Gate)
The T-Gate applies a **π/4 phase shift** to the qubit. It is essential for quantum computing because it, along with the Hadamard and CNOT gates, allows for **universal quantum computation**. Mathematically:

$$T|0\rangle = |0\rangle$$
$$T|1\rangle = e^{i\pi/4} |1\rangle$$

## CSWAP Gate (Controlled-SWAP / Fredkin Gate)
The CSWAP gate, also known as the **Fredkin gate**, is a three-qubit gate that conditionally swaps the states of two target qubits based on the state of a control qubit. If the control qubit is in the $|1\rangle$ state, it swaps the states of the two target qubits; otherwise, it leaves them unchanged.

### Mathematical Definition

The CSWAP gate acts on three qubits: a control qubit $|c\rangle$ and two target qubits $|t_1\rangle$ and $|t_2\rangle$. The operation is:

$$\text{CSWAP}|c\rangle|t_1\rangle|t_2\rangle = \begin{cases} |c\rangle|t_1\rangle|t_2\rangle & \text{if } c = 0 \\ |c\rangle|t_2\rangle|t_1\rangle & \text{if } c = 1 \end{cases}$$

In matrix form (for the 8-dimensional space of three qubits), the CSWAP gate is:

$$\text{CSWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{pmatrix}$$

The CSWAP gate is fundamental in quantum algorithms such as the swap test, quantum error correction, and quantum state comparison. The CSWAP gate is reversible and preserves the number of $|1\rangle$ states in the system (conserves the Hamming weight).

## U Gate (Universal Single-Qubit Gate)
The U gate is a **universal single-qubit gate** parameterized by three angles ($\theta$, $\phi$, $\lambda$) that can represent any single-qubit unitary operation. It provides a complete parameterization of single-qubit rotations and is essential for implementing arbitrary quantum operations.

### Mathematical Definition

The U gate matrix representation is:

$$U(\theta, \phi, \lambda) = \begin{pmatrix} \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\ e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2) \end{pmatrix}$$

The U gate can be decomposed into rotations around the Z, Y, and Z axes:

$$U(\theta, \phi, \lambda) = R_z(\phi) \cdot R_y(\theta) \cdot R_z(\lambda)$$

This decomposition shows that the U gate applies:
1. A rotation by λ around the Z-axis
2. A rotation by θ around the Y-axis
3. A rotation by φ around the Z-axis

### Special Cases

- **Identity**: U(0, 0, 0) = I
- **Pauli X**: U(π, 0, π) = X
- **Pauli Y**: U(π, π/2, π/2) = Y
- **Pauli Z**: U(0, 0, π) = Z
- **Hadamard**: U(π/2, 0, π) = H

This gate is particularly useful in parameterized quantum circuits and variational quantum algorithms where you need to optimize over all possible single-qubit operations.

# **Updates**
- **Acknowledged support for Cirq & Braket** (New Addition)
- **Removed Pauli X Gate** (Merged into NOT Gate)
- **Added T-Gate** (New Addition)
- **Added CSWAP Gate** (New Addition)
- **Added U Gate** (New Addition)
- **Fixed typos**

These quantum gates are fundamental building blocks in quantum computing, enabling the manipulation and transformation of qubit states to perform various quantum algorithms and computations.
