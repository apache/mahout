---
title: Pqc
---
# Parameterized Quantum Circuits: Developer's Guide

Welcome to the guide designed for developers interested in implementing parameterized quantum circuits (PQCs) from scratch. This document provides detailed information about the theory, design, and applications of PQCs without reliance on existing quantum computing frameworks like Qiskit or Cirq.

---

## Contents

1. **Introduction to Parameterized Quantum Circuits**
2. **Variational Quantum Algorithms**
3. **Designing Parameterized Quantum Circuits**
4. **Training Parameterized Quantum Circuits**
5. **Quantum Machine Learning with PQCs**
6. **Optimization Challenges and Solutions**
7. **Noise and Error Mitigation in PQCs**
8. **Expressibility and Entanglement in PQCs**
9. **Hardware Considerations and Implementations**
10. **Advanced Applications of PQCs**
11. **Scalability and Resource Estimations**
12. **Current Research and Open Problems**

---

## 1. Introduction to Parameterized Quantum Circuits

### Quantum Gates and Circuits

Quantum gates are the building blocks of quantum circuits, functioning similarly to classical logic gates but operating on qubits, the fundamental units of quantum information. Unlike classical bits, which can be either 0 or 1, qubits can exist in superpositions of states, represented as linear combinations of |0⟩ and |1⟩.

**Universal Gate Sets:** To perform arbitrary quantum computations, it's essential to understand universal gate sets. A set of quantum gates is considered universal if any unitary operation (any quantum computation) can be approximated to arbitrary accuracy using these gates. Common universal gate sets include the set of Hadamard (H), CNOT, and $T$ gates. The H gate creates superpositions, CNOT is an entangling operation, and the $T$ gate provides a necessary phase shift.

### Parameterized Gates

Parameterized gates are quantum gates that include parameters that can be adjusted, often continuously. These gates are integral to parameterized quantum circuits, as their parameters can be optimized during algorithms. Common parameterized gates include rotation gates such as $R_x(\theta)$, $R_y(\theta)$, and $R_z(\theta)$, each corresponding to a rotation around the respective axes in the Bloch sphere representation of a qubit.

**Rotation Gates:** For instance, a rotation gate around the x-axis, $R_x(\theta)$, can be represented by the unitary operation:
$$R_x(\theta) = \exp\left(-i \frac{\theta}{2} \sigma_x\right)$$
where $\sigma_x$ is the Pauli X matrix. The parameter $\theta$ can be adjusted during the circuit's operation, which is key in variational and learning-based quantum algorithms.

### Motivation for PQCs

The motivation for using parameterized quantum circuits (PQCs) rests in their adaptability and efficiency in capturing complex quantum phenomena. PQCs are a powerful tool for near-term quantum computing, particularly in approaches like hybrid quantum-classical algorithms where classical optimizers adjust quantum circuit parameters to minimize error or energy functions.

**Near-Term Applications:** PQCs are pivotal in the realm of Noisy Intermediate-Scale Quantum (NISQ) technology, where current quantum hardware limitations require algorithms that do not require fault-tolerant quantum error correction. They are used extensively in variational quantum algorithms, where quantum resources are utilized to process information, and classical algorithms handle the parameter optimization.

By understanding these foundational concepts, developers can implement software architectures that accurately simulate and manipulate parameterized quantum circuits, forming a basis for advanced quantum computational applications.

## 2. Variational Quantum Algorithms (VQAs)

Variational Quantum Algorithms (VQAs) leverage the principles of both quantum mechanics and classical optimization to solve complex problems efficiently. They are central to near-term applications of quantum computing because of their ability to work within the constraints of noisy intermediate-scale quantum (NISQ) devices.

### Variational Principle

The variational principle is a foundational concept in quantum mechanics, especially useful for approximating the ground state energy of a quantum system. The basic idea is that the lowest energy an electron in a molecule can have is governed by this principle. By constructing a parameterized quantum circuit to prepare various quantum states that serve as trial solutions, VQAs explore the energy landscape to find the state that minimizes a given cost function, typically expected energy.

### Algorithm Examples

- **Variational Quantum Eigensolver (VQE):** A VQA designed to find the eigenvalues of a Hamiltonian, which is crucial in quantum chemistry for determining molecule energy levels. VQE encodes the Hamiltonian of a molecule into a PQC. The circuit parameters are tuned using a classical optimizer to minimize the expectation value of the Hamiltonian, thereby approximating the molecule’s ground state energy.
- **Quantum Approximate Optimization Algorithm (QAOA):** This algorithm is used for solving combinatorial optimization problems. QAOA represents problem constraints as Hamiltonians and finds the bit string that minimizes these constraints. The algorithm uses a PQC to approximate the solution state, systematically optimizing its parameters to improve the solution quality.

### Role of PQCs

In the context of VQAs, PQCs act as an approximation model which can be adjusted through parameter tuning. Unlike traditional quantum algorithms that require a precise sequence of quantum operations, PQCs provide a flexible framework where parameters can be continuously varied. This adaptability makes them well-suited for iterative optimization processes essential in VQAs.

PQCs leverage parameterized gates capable of representing a wide class of quantum states. Through classical-quantum interplay, where quantum circuits process the data while classical systems optimize the parameters based on the feedback, the PQCs are tuned to solve specific problems or approximate desired states. This makes PQCs integral to enabling VQAs to achieve high fidelity results while remaining viable on NISQ devices.

Overall, VQAs represent one of the most promising approaches to practical quantum advantage. They serve as a bridge from current hardware capabilities to solving meaningful problems, with PQCs being the core mechanism allowing flexibility, adaptability, and execution within noisy constraints.

## 3. Designing Parameterized Quantum Circuits

### Key Concepts:

#### Circuit Ansätze

A variational quantum circuit or ansatz is a crucial part of designing parameterized quantum circuits. The design of these circuits can be oriented towards hardware efficiency, problem inspiration, or heuristic patterns:

- **Hardware-Efficient Ansätze:** These are designed to work well on existing quantum devices, minimizing the circuit depth and using gates that are easily implementable on specific quantum hardware. For example, these ansätze might use a set of parameterized single-qubit rotation gates and CNOTs pre-optimized for the target hardware's fidelity and connectivity. The main advantage is reduced error rates due to shorter circuits and fewer gate operations.

- **Problem-Inspired Ansätze:** Tailored to exploit the structure of the specific problem at hand, these ansätze draw on insights from the problem's domain, such as quantum chemistry or optimization. For instance, in VQE, one might use ansätze inspired by known efficient classical approximations or simplified physical models of the molecular system.

- **Heuristic Ansätze:** These are not derived from specific physical insights but instead rely on general properties like expressibility and trainability. These might include layered structures where each layer applies the same set of gates. This flexibility can help in exploring a wider state space to find potentially optimal quantum solutions.

#### Entanglement Schemes

Entanglement is a fundamental resource in quantum computing, and how it is introduced in a circuit impacts its efficacy:

- An appropriate entanglement scheme ensures that the ansatz can explore complex, entangled states of the system, which are essential for solving many quantum problems efficiently. Common schemes involve multi-qubit gate layers like CNOT gates or controlled-Z gates applied in a specific pattern to ensure all qubits are appropriately correlated.

- Balancing the 'amount' and 'distribution' of entanglement is non-trivial and often problem-specific. Too little entanglement might make the circuit unable to represent the required state complexity, whereas excessive entanglement could lead to optimization challenges, such as barren plateaus.

#### Depth vs. Expressibility Trade-offs

When designing parameterized quantum circuits, a critical challenge is balancing circuit depth and expressibility:

- **Circuit Depth:** A deeper circuit can potentially represent more complex states but also incurs higher noise and decoherence in real quantum hardware. The depth must stay within the coherence time limits of the quantum device to prevent excessive error accumulation.

- **Expressibility:** This refers to the circuit’s ability to cover a vast swath of the Hilbert space. More expressibility generally means the circuit can represent a broader range of quantum states, thereby solving a more extensive set of problems or finding better approximations.

The trade-offs must be carefully considered when designing circuits, especially in hardware where decoherence and gate error rates limit the practical depth of any quantum circuit.

These design principles are foundational for creating efficient, effective parameterized quantum circuits that meet the demands of specific applications, striking a fine balance between theoretical expressibility and practical implementability.

## 4. Training Parameterized Quantum Circuits

**Key Concepts:**

### Optimization Methods

The training of parameterized quantum circuits (PQCs) involves optimizing the parameters of the quantum gates to minimize a specific cost function, typically corresponding to the expectation value of an observable. The two main classes of optimization methods are:

- **Gradient-Based Methods:**
  These methods use derivatives of the cost function with respect to the circuit parameters to navigate the optimization landscape. The **parameter shift rule** is particularly important, as it provides a way to estimate gradients analytically on quantum hardware. The rule leverages the circuit's periodic nature to compute the gradient by evaluating the cost at a small number of shifted parameter values. Common gradient-based techniques include:

  - **Gradient Descent**: Iteratively updates parameters in the direction of the negative gradient.
  - **Adam**: An adaptive learning rate optimization algorithm that combines momentum and scaling techniques.

- **Gradient-Free Methods:**
  These approaches do not require explicit gradient computation and are useful when gradients are costly or infeasible to obtain. Methods like **Nelder-Mead** and **COBYLA** evaluate the cost function at different parameter settings to guide optimization. These methods are often more resilient to noise, making them suitable for noisy quantum devices.

### Barren Plateaus

**Barren plateaus** refer to regions in the parameter space where the gradient is extremely small, leading to slow convergence of the optimization algorithms. Several factors contribute to barren plateaus:

- **Circuit Depth**: As circuit depth increases, gradients tend to vanish. Balancing depth with expressibility is crucial.
- **Random Initialization**: Arbitrary parameter initialization can lead to initial points in barren plateaus. Careful initialization strategies are necessary to circumvent this issue.

Mitigation strategies include using circuit architectures tailored to specific problems or employing progressive, layer-wise training to narrow down the parameter space iteratively.

### Parameter Shift Rule

The **parameter shift rule** is an essential tool for obtaining exact gradients of parameters in a PQC. It is specifically designed taking into account the unitary nature of quantum gates. For a parameterized gate $U(\theta)$, the gradient of the expectation value $\langle \psi | U^{\dagger}(\theta) O U(\theta) | \psi \rangle$ with respect to $\theta$ can be calculated by evaluating the expectation with shifted parameters:

$$\frac{d}{d\theta} \langle \psi | U^\dagger(\theta) O U(\theta) |\psi\rangle = \frac{1}{2}\left(\langle \psi | U^\dagger(\theta + \frac{\pi}{2}) O U(\theta + \frac{\pi}{2}) |\psi\rangle - \langle \psi | U^\dagger(\theta - \frac{\pi}{2}) O U(\theta - \frac{\pi}{2}) |\psi\rangle\right)$$

This approach is hardware-efficient since it requires only a few additional circuit evaluations per parameter.

---

These key concepts provide foundational insights into the training procedures for PQCs, emphasizing the careful selection of optimization strategies, understanding the challenges with barren plateaus, and leveraging theoretical methods like the parameter shift rule to achieve effective parameter updates.

## 5. Quantum Machine Learning with PQCs

**Quantum Classifiers**

Quantum classifiers are used in machine learning to categorize or predict outcomes based on quantum-encoded data. Leveraging PQCs, these classifiers can offer novel approaches to traditional classification tasks by exploiting quantum superposition and entanglement. Unlike classical classifiers, quantum classifiers can potentially offer an exponential increase in feature space, which is particularly advantageous for complex datasets. A typical quantum classifier involves feeding classical data into a quantum circuit, where it is processed by a parameterized circuit before being measured. The outcomes of these measurements are used to make predictions or classifications, often through a post-processing classical algorithm that determines the final output based on quantum results.

**Data Encoding Strategies**

To make use of quantum classifiers, efficient data encoding is crucial:

- **Amplitude Encoding** utilizes the amplitudes of quantum states to encode classical data. This method can encode $2^n$ dimensional data into n qubits, offering exponential data compression. However, the challenge lies in preparing the exact quantum state, which can be resource-intensive and sensitive to noise.

- **Angle Encoding** involves mapping classical data points directly to the rotation angles of quantum gates. This simple yet effective method can encode data by transforming features into angles that control the operations of parameterized gates (e.g., RX, RY, RZ). Its advantage lies in the straightforward implementation and flexibility, but it may not fully exploit the exponential scaling potential.

Quantum Machine Learning models often use these encoding strategies to input data into quantum circuits, which, in combination with different PQC designs, perform a wide range of quantum-enhanced computations. The effectiveness of the encoding strategy is context-dependent and often dictates the potential advantage over classical systems.

**Quantum Neural Networks (QNNs)**

Quantum Neural Networks represent a new paradigm inspired by classical neural networks, combining quantum computing principles with deep learning architectures. At their core, QNNs harness parameterized quantum circuits as quantum layers interspersed with classical processing layers. The ability of quantum circuits to span highly complex state spaces offers an avenue for significant representation power.

A QNN typically includes a classical dataset, which is encoded into quantum states and processed through several layers involving parameterized gates. Each layer adjusts parameters during training, akin to updating weights in classical neural networks. QNNs can potentially leverage both classical computing's strengths through hybrid symbiosis and quantum advantages like entanglement and superposition, potentially solving problems intractable by classical counterparts.

**Key Considerations in Implementing Quantum ML Models:**

1. **Scalability:** Current quantum hardware has limitations on qubit count and coherence time. Ensuring circuits are designed with hardware constraints in mind is essential. Scaling QNNs is a non-trivial task and requires careful balancing between depth (and corresponding expressibility) and noise mitigation.

2. **Noisy Intermediate-Scale Quantum (NISQ) Presence:** Tailoring QNN architectures to operate effectively on NISQ devices, which are susceptible to environmental noise, involves employing error mitigation and robust training techniques.

3. **Hybrid Frameworks:** Often QNNs are employed in hybrid frameworks where quantum computation handles certain tasks (e.g., feature space transformations), while the bulk of data manipulation occurs classically. Understanding where quantum processing provides advantages is pivotal for designing effective hybrid solutions.

This section provides a gateway into quantum-enhanced machine learning applications, highlighting critical considerations and emerging approaches in the field through parameterized quantum circuits. The distinct nature of quantum mechanics offers exciting possibilities for future developments in machine learning paradigms.

## 6. Optimization Challenges and Solutions

### Mitigating Barren Plateaus

Barren plateaus present significant challenges in optimizing parameterized quantum circuits (PQCs). These are regions in the parameter space where the gradients of the cost function vanish, making it difficult to locate the optimal parameter settings using gradient descent methods. The presence of barren plateaus often increases with the size and complexity of the quantum circuit, posing a significant hurdle for scalability.

#### Strategies to Mitigate Barren Plateaus:

1. **Layerwise Training:** This approach involves training the quantum circuit in layers or segments rather than as a whole. By optimizing each segment individually, one can manage the complexity of the optimization process, thereby reducing the likelihood of encountering barren plateaus.

2. **Random Initialization:** During initialization, parameters can be set randomly in a way that avoids symmetries in the parameter landscape that can lead to flat regions. Using prior knowledge to smartly choose these initial parameters can break symmetries that contribute to barren plateaus.

3. **Adaptive Learning Rates:** Dynamically adjusting the learning rates during training can help navigate around or out of barren plateaus. Low gradients can be handled by slower learning rates that allow for more precise searching of the parameter space.

4. **Heuristic Approaches:** Utilize heuristic or problem-specific information to bias the initialization of parameters or the design of the circuit ansatz to regions of the parameter space with favorable optimization landscapes.

### Noise-Induced Difficulties

Noise in quantum circuits can exacerbate the challenges posed by barren plateaus, as it can obscure the signal required for successful optimization. Noise originates from hardware imperfections and environmental factors that compromise the accuracy of quantum operations.

#### Mitigation Techniques:

1. **Noise-Resilient Cost Functions:** Designing cost functions that are less sensitive to noise can help maintain optimization progress even in the presence of significant hardware noise.

2. **Error Mitigation Strategies:** Techniques like error extrapolation or error correction can be employed to reduce the impact of noise on the optimization process. These methods aim to "simulate" a noise-free environment or directly correct errors introduced by noise.

3. **Robust Circuit Design:** Developing circuits that inherently minimize the effective noise through error-resistant configurations or operational strategies keeps optimization signals strong.

4. **Use of Quantum Simulators for Training:** Simulators can provide an environment to pre-train circuits, allowing them to reach a certain level of optimization before being deployed to real hardware where noise is a factor.

These optimization challenges are fundamental to the development and deployment of effective PQCs, especially in the currently noise-prone quantum computers. By understanding and addressing these issues, developers can significantly enhance the performance and scalability of PQCs in real-world applications.

## 7. Noise and Error Mitigation in PQCs

**Key Concepts:**

- **Hardware Noise**

  Hardware noise is an inherent challenge in quantum computing that affects the reliability and accuracy of quantum computations. Types of noise include decoherence, gate errors, readout errors, and crosstalk, each of which can degrade the performance of Parameterized Quantum Circuits (PQCs). Understanding these noise sources is critical for developing techniques to mitigate their effects. Decoherence refers to the loss of quantum coherence in qubits due to interaction with their environment. Gate errors occur due to imperfections in implementing quantum gates. Readout errors happen during the measurement process, resulting in incorrect qubit states being recorded. Crosstalk involves unintended interactions between qubits or quantum gates, complicating the execution of precise quantum operations. Identifying these errors and modeling their impact help in developing robust mitigation strategies.

- **Error Mitigation Techniques**

  Error mitigation is crucial for enhancing the accuracy of PQCs without the overhead of full quantum error correction, which is not feasible on current noisy intermediate-scale quantum (NISQ) devices. Popular techniques include:

  1. **Zero-Noise Extrapolation:** This approach involves executing the quantum circuit at different noise levels and extrapolating the results to an estimated zero-noise scenario. This can be achieved by intentionally increasing the noise through techniques like gate repetition and using polynomial or linear extrapolation methods to predict results as if no noise were present.

  2. **Probabilistic Error Cancellation:** This method involves creating an error model to characterize the types of noise affecting the circuit. A quantum operation is devised to effectively invert this noise in a probabilistic manner, canceling out errors in expected outcomes. Calculating the inverse operations requires an accurate noise model and efficient computation of the noise-free probability utilizing classical post-processing techniques.

These error mitigation strategies are vital for making PQCs feasible on current quantum hardware, allowing developers to improve the fidelity of quantum operations and obtain reliable results even in the presence of significant noise. Employing these techniques enables practical applications of PQCs, such as variational quantum algorithms and quantum machine learning, on NISQ devices.

## 8. Expressibility and Entanglement in PQCs

**Key Concepts:**

- **Circuit Expressibility:**

  Expressibility in parameterized quantum circuits (PQCs) refers to the ability of a circuit to represent a wide range of quantum states. This is crucial for ensuring that the quantum circuit can span enough of the state space to solve a given problem effectively. When designing a PQC, one must consider whether the chosen ansatz can adequately represent the required state with the available quantum resources. The expressibility can sometimes be quantified using metrics such as the concentration of measure, which evaluates how uniformly the circuit covers the state space over different parameter configurations.

  A highly expressible circuit might not always be desirable, as it could also introduce complexities like barren plateaus—regions in the parameter space where the gradient becomes very small, making optimization difficult. Therefore, developers need to balance expressibility with trainability. This may involve choosing or designing ansätze that are inherently structured to target particular types of problems, leveraging prior knowledge about the problem domain to select more efficient pathways through parameter space.

- **Entanglement Measures:**

  Entanglement is a fundamental resource in quantum computing, enabling the powerful computational capabilities of quantum systems. In the context of PQCs, the degree of entanglement produced by a circuit is often aligned with its potential computational power. Entangled states are necessary for many quantum algorithms that offer a speedup over classical ones.

  To evaluate entanglement within a PQC, one can use various measures, such as the entanglement entropy, which quantifies the degree of entanglement between different parts of a quantum system. A developer must ensure that the circuit design incorporates sufficient entangling operations to leverage this quantum resource effectively. However, similar to expressibility, there's a trade-off involved. Excessive entanglement may lead to increased susceptibility to noise and other decoherence effects in real quantum hardware, especially in noisy intermediate-scale quantum (NISQ) devices, where noise levels are non-negligible.

  In designing a PQC, considering the connectivity of qubits on the target hardware is crucial, as hardware limitations may restrict which qubits can be directly entangled with one another. This necessitates efficient mapping strategies that translate the ideal circuit design into one that respects these constraints while still achieving desired levels of entanglement.

## 9. Hardware Considerations and Implementations

**Gate Fidelities and Connectivity**

The fidelity of quantum gates is a crucial factor in quantum computing. Fidelity measures how accurately a quantum gate performs the operation it is intended to implement. High-fidelity gates are essential to ensure that the quantum computations are reliable and less prone to errors. In practice, gate errors can arise from various sources like cross-talk, decoherence, or imperfect calibration. Developers must design parameterized quantum circuits (PQCs) with these constraints in mind.

Connectivity in quantum hardware pertains to the ability of qubits to interact with one another directly. Not all qubits in a quantum device can be entangled with each other due to specific connectivity graphs set by architecture. This limitation affects the design of PQCs as it determines which qubits can easily share entangled states or swapped gate operations. Efficiently mapping a logical circuit with qubits that optimally fit the connectivity map involves the use of SWAP gates at strategic locations to mitigate the issue, which can otherwise lead to increased circuit depth and further introduce errors.

**Pulse-Level Control**

Pulse-level control refers to the fine-tuning of the microwave or laser pulses that drive quantum gates in a physical quantum system. When qubits are manipulated with pulses that create a desired change in their states, the precision of these operations is paramount. Developers designing PQCs from scratch may delve into pulse calibration and shaping techniques to optimize quantum gate operations. Understanding how pulse distortions affect gate fidelity, and learning how to correct them, allows developers to improve PQC implementations' performance, both in terms of speed and accuracy.

Pulse-level control also opens opportunities for new optimization strategies such as designing custom gates that may offer better performance for certain PQC tasks. By customizing pulse sequences, developers can circumvent some of the inefficiencies present in standard gate operations, thus achieving higher fidelity and reduced execution time. Mastery over pulse control can lead to enhancements in how variational algorithms perform under real-world conditions, where hardware noise and decoherence significantly affect outcomes.

## 10. Advanced Applications of PQCs

In this section, we delve into some of the cutting-edge applications of parameterized quantum circuits (PQCs) across various fields of research and industry. By exploring these advanced applications, developers can gain insight into the potential of PQCs to solve real-world problems and drive innovation in computing.

### Quantum Generative Models

Parameterized quantum circuits play a vital role in the domain of quantum generative models. These models, such as Quantum Generative Adversarial Networks (QGANs) and Born Machines, leverage quantum superposition and entanglement to generate complex probabilistic distributions that classical models struggle to replicate.

- **Quantum GANs (QGANs):** Based on the classical GAN architecture, QGANs comprise a generator and a discriminator implemented via PQCs. The generator circuit produces quantum states intended to mimic the true data distribution, while the discriminator evaluates the authenticity of the quantum samples. This adversarial setup allows QGANs to potentially surpass classical generative models in efficiency and performance, particularly for data types where quantum data encoding offers a natural advantage.

- **Born Machines:** Inspired by the probabilistic interpretation of quantum mechanics, Born machines utilize PQCs to define probability distributions through the Born rule. By appropriately setting parameters, these circuits can model complex distributions and have applications in areas such as unsupervised learning and probabilistic inference.

### Quantum Chemistry and Material Science

One of the significant applications of PQCs is in quantum chemistry and material science, where these circuits address the computational challenges of simulating molecular and atomic interactions.

- **Molecule Simulation:** PQCs are used in algorithms like the Variational Quantum Eigensolver (VQE) to approximate the electronic structure (ground state energy) of molecules. By designing problem-inspired ansätze, PQCs efficiently capture molecular properties, potentially leading to breakthroughs in understanding chemical reactions, discovering new materials, and optimizing catalysts.

- **Quantum Phase Estimation:** While traditionally considered a resource-intensive task, PQCs employed in variational forms provide a hybrid approach to quantum phase estimation. This influences study areas such as superconductivity and correlated electron systems by offering insights into their energy landscapes and phase structures.

### Quantum Finance and Portfolio Optimization

The financial industry is another field ripe for innovation through PQCs, where they offer novel methods for risk assessment, asset pricing, and portfolio optimization.

- **Risk Analysis:** PQCs can model complex financial systems more naturally than classical algorithms by exploiting superposition and entanglement to analyze multiple risk scenarios concurrently. This capability could lead to more accurate predictions and better risk management strategies.

- **Portfolio Optimization:** Using quantum optimization algorithms, such as QAOA with PQCs, investors can potentially solve portfolio optimization problems more efficiently than classical methods, taking into account a multitude of constraints and objectives that characterize financial decisions.

By understanding these advanced applications, developers can conceptualize and implement parameterized quantum circuits in groundbreaking ways, pushing the boundaries of current technological capabilities. As quantum hardware continues to evolve, the scope and impact of PQCs in diverse scientific and commercial sectors are poised to expand dramatically.
