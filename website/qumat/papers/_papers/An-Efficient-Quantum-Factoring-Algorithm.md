---
layout: post
title: "Summary of 'An Efficient Quantum Factoring Algorithm'"
date: 2024-03-04
---

Author: Oded Regev

[Original Paper](https://arxiv.org/abs/2308.06572)

The paper presents an efficient quantum factoring algorithm that can be used to
factorize n-bit integers. The algorithm involves running a quantum circuit with
˜O(n3/2) gates for √n + 4 times, and then using a polynomial-time classical
post-processing step. The correctness of the algorithm is based on a
number-theoretic assumption similar to those used in subexponential classical
factorization algorithms. The author demonstrates that quantum circuits of size
˜O(n3/2) are sufficient for factoring integers, which is an improvement over
previous algorithms that required larger circuit sizes. The number of qubits in
the quantum circuit is O(n3/2), which is higher than the qubit requirement in
optimized implementations of Shor's algorithm. However, the depth of the quantum
circuit is smaller than Shor's algorithm, making it more feasible for
implementation. The paper also discusses the potential implications of the
algorithm in practice. It is highlighted that the analysis is asymptotic and the
algorithm may not be efficient for small values of n. The algorithm may benefit
from optimizations in fast integer multiplication and the use of smaller qubit
counts, similar to optimizations used in Shor's algorithm. However, it is
currently unclear if these optimizations can be applied to the proposed
algorithm. The author concludes by stating that the algorithm provides an
improvement over Shor's algorithm in terms of circuit size. However, it remains
to be seen if the algorithm can be practically implemented and if it can provide
an improvement over Shor's algorithm for small values of n. The analysis in the
paper is based on asymptotics, and it is unclear if hidden constants in the
algorithm would make it inefficient for small values of n. In summary, the paper
presents an efficient quantum factoring algorithm that uses a quantum circuit
with ˜O(n3/2) gates and a classical post-processing step. The algorithm provides
an improvement over previous algorithms in terms of circuit size, but its
practicality and potential improvements for small values of n remain to be seen.
