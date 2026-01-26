---
title: "Summary of 'Quantum Kernel Estimation With Neutral Atoms For Supervised Classification: A Gate-Based Approach'"
date: 2024-03-06
---

Author: Marco Russo, Edoardo Giusto, Bartolomeo Montrucchio

[Original Paper](https://arxiv.org/abs/2307.15840)

In this paper, the authors propose a gate-based approach to quantum kernel
estimation (QKE) for supervised classification using neutral atom quantum
computers. QKE is a technique that leverages the power of quantum computing to
estimate a kernel function that is difficult to compute classically. The
estimated kernel is then used by a classical computer to train a support vector
machine (SVM) for classification tasks. The authors focus on neutral atom
quantum computers because they allow for more freedom in arranging the atoms,
which is essential for implementing the necessary gates for QKE. They present a
general method for deriving 1-qubit and 2-qubit gates from laser pulses, which
are then used to construct a parameterized sequence for feature mapping on 3
qubits. They show that this approach can be extended to N qubits, taking
advantage of the more flexible arrangement of atoms in neutral atom devices. The
experimental setup involves simulating the Pasqal Chadoq2 device, which allows
for planar arrangement of atoms. The authors generate a dataset of 40 training
samples and 20 test samples with 3 features and a separation gap of 0.1. They
use the Qiskit library to implement the feature mapping circuit and generate the
sequences of pulses for QKE. The training and testing of the SVM are performed
on a classical computer using the estimated kernel matrices. The results show
that the proposed approach achieves a high accuracy of 75% on the test set,
despite the small size of the dataset and the low separation. The authors
compare the performance to a classical SVM with a radial basis function kernel
and find that the quantum approach outperforms the classical approach. The
authors discuss the advantages of using neutral atom quantum computers for QKE.
The arbitrary arrangement of atoms allows for more direct connections between
qubits, reducing the depth of the circuit and reducing the impact of
decoherence. They also highlight the exponential computational advantage of
quantum feature kernels over classical kernel computation methods for
high-dimensional feature spaces. Overall, the paper presents a gate-based
approach to QKE using neutral atom quantum computers. The experimental results
demonstrate the potential of this approach for supervised classification tasks
and highlight the advantages of neutral atom devices for implementing QKE
circuits. The paper provides a foundation for future research in the field of
quantum machine learning and quantum computing.
