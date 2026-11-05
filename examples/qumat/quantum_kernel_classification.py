#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Classify a toy dataset with a quantum kernel.

The quantum computer (here a simulator) only produces the kernel matrix.
Training and prediction are ordinary classical kernel methods:

1. Kernel ridge classification, written in a few lines of NumPy, which
   always runs.
2. scikit-learn's ``SVC(kernel="precomputed")``, only if scikit-learn is
   installed (it is not a Qumat dependency).

Run from the repository root::

    python examples/qumat/quantum_kernel_classification.py
"""

import numpy as np

from qumat import AngleFeatureMap, ZZFeatureMap, kernel_matrix

BACKEND_CONFIG = {
    "backend_name": "qiskit",
    "backend_options": {"simulator_type": "aer_simulator", "shots": 1024},
}


def make_dataset(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two noisy clusters in [0, pi]^2 with labels -1 and +1."""
    rng = np.random.default_rng(seed)
    a = rng.normal([0.8, 0.8], 0.35, size=(n_per_class, 2))
    b = rng.normal([2.3, 2.3], 0.35, size=(n_per_class, 2))
    x = np.clip(np.vstack([a, b]), 0.0, np.pi)
    y = np.array([-1] * n_per_class + [1] * n_per_class)
    return x, y


def kernel_ridge_fit(k_train: np.ndarray, y: np.ndarray, reg: float) -> np.ndarray:
    """Solve (K + reg * I) alpha = y for the dual coefficients alpha."""
    return np.linalg.solve(k_train + reg * np.eye(len(y)), y)


def evaluate(name: str, feature_map, data) -> None:
    x_train, y_train, x_test, y_test = data

    # Training Gram matrix: N x N, symmetric, ones on the diagonal.
    k_train = kernel_matrix(
        x_train, feature_map=feature_map, backend_config=BACKEND_CONFIG
    )
    # Test-vs-train matrix: rows are test points, columns training points.
    k_test = kernel_matrix(
        x_test, x_train, feature_map=feature_map, backend_config=BACKEND_CONFIG
    )
    print(f"[{name}] train kernel {k_train.shape}, test kernel {k_test.shape}")
    print(
        f"[{name}] min eigenvalue of K_train: {np.linalg.eigvalsh(k_train).min():.1e}"
    )

    alpha = kernel_ridge_fit(k_train, y_train, reg=0.1)
    pred = np.sign(k_test @ alpha)
    print(f"[{name}] kernel ridge accuracy: {np.mean(pred == y_test):.2f}")

    try:
        from sklearn.svm import SVC  # ty: ignore[unresolved-import]
    except ImportError:
        print(f"[{name}] scikit-learn not installed; skipping SVC")
        return
    svc = SVC(kernel="precomputed").fit(k_train, y_train)
    print(
        f"[{name}] SVC(kernel='precomputed') accuracy: {svc.score(k_test, y_test):.2f}"
    )


def main() -> None:
    x_train, y_train = make_dataset(n_per_class=8, seed=7)
    x_test, y_test = make_dataset(n_per_class=5, seed=11)
    data = (x_train, y_train, x_test, y_test)

    # The angle map's kernel is a smooth similarity (prod cos^2 of feature
    # differences), which suits two compact clusters well.
    evaluate("angle", AngleFeatureMap(num_qubits=2), data)

    # The ZZ map is periodic in each feature and was designed for data
    # with that structure (Havlicek et al. 2019). On plain clusters it does
    # noticeably worse -- a reminder that the feature map is a modelling
    # choice, not a free quantum advantage.
    evaluate("zz", ZZFeatureMap(num_qubits=2, reps=1), data)


if __name__ == "__main__":
    main()
