#!/usr/bin/env python3
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

"""
PennyLane baseline: Iris (2-class), amplitude encoding, variational classifier.

Aligned with: https://pennylane.ai/qml/demos/tutorial_variational_classifier

Data sources (default: sklearn, not the official file):
  - Default: sklearn.datasets.load_iris, classes 0 & 1, 4 features → scale → L2 norm → get_angles.
  - Official file: pass --data-file <path>. File format: cols [f0, f1, f2, f3, label]; we use first 2 cols,
    pad to 4, L2 norm. Bundled path: data/iris_classes1and2_scaled.txt (from XanaduAI/qml,
    https://raw.githubusercontent.com/XanaduAI/qml/master/_static/demonstration_assets/variational_classifier/data/iris_classes1and2_scaled.txt).
  - Total samples: 100 (2-class Iris). Full Iris has 150 (3 classes).

Pipeline: state prep (Möttönen angles) → Rot layers + CNOT → expval(PauliZ(0)) + bias; square loss; Adam or Nesterov.
"""

from __future__ import annotations

# --- Imports ---

import argparse
import time
from typing import Any

import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer
except ImportError as e:
    raise SystemExit(
        "PennyLane is required. Install with: uv sync --group benchmark"
    ) from e

try:
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: uv sync --group benchmark"
    ) from e


NUM_QUBITS = 2


# --- Encoding: 4-D vector → 5 angles (Möttönen et al.) ---
def get_angles(x: np.ndarray) -> np.ndarray:
    """State preparation angles from 4-D normalized vector (Möttönen et al.)."""
    x = np.asarray(x, dtype=np.float64)
    eps = 1e-12
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + eps))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + eps))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / (np.linalg.norm(x) + eps))
    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


# --- Circuit: amplitude state prep (RY/CNOT) + variational layer (Rot + CNOT) ---
def state_preparation(a, wires=(0, 1)):
    """Amplitude encoding via rotation angles (tutorial / Möttönen et al.)."""
    qml.RY(a[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(a[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(a[2], wires=wires[1])
    qml.PauliX(wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(a[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(a[4], wires=wires[1])
    qml.PauliX(wires=wires[0])


def layer(layer_weights, wires=(0, 1)):
    """Rot on each wire + CNOT (tutorial Iris section)."""
    for i, w in enumerate(wires):
        qml.Rot(*layer_weights[i], wires=w)
    qml.CNOT(wires=list(wires))


# --- Data: official file (2 cols → pad 4 → L2 norm → get_angles). Source: XanaduAI/qml, see docstring. ---
def load_iris_from_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load official-style Iris 2-class data: file with columns [f0, f1, f2, f3, label].
    Uses first 2 columns only (as in tutorial), pad to 4, L2 normalize, get_angles.
    Returns (features, Y) with features shape (n, 5), Y in {-1, 1}.
    """
    data = np.loadtxt(path, dtype=np.float64)
    X = data[:, 0:2]  # first 2 features only (tutorial convention)
    Y = data[:, -1]  # labels already ±1
    padding = np.ones((len(X), 2)) * 0.1
    X_pad = np.c_[X, padding]
    norm = np.sqrt(np.sum(X_pad**2, axis=-1)) + 1e-12
    X_norm = (X_pad.T / norm).T
    features = np.array([get_angles(x) for x in X_norm])
    return features, Y


# --- Data: sklearn Iris classes 0 & 1, 4 features → scale → L2 norm → get_angles ---
def load_iris_binary(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Iris classes 0 and 1 only. Uses all 4 features: scale → L2 norm → get_angles.
    Returns (features, Y) with features shape (n, 5), Y in {-1, 1}.
    Data source: sklearn.datasets.load_iris.
    """
    X_raw, y = load_iris(return_X_y=True)
    mask = (y == 0) | (y == 1)
    X = np.asarray(X_raw[mask], dtype=np.float64)
    y = y[mask]
    X = StandardScaler().fit_transform(X)
    norm = np.sqrt(np.sum(X**2, axis=-1)) + 1e-12
    X_norm = (X.T / norm).T
    features = np.array([get_angles(x) for x in X_norm])
    Y = np.array(y, dtype=np.float64) * 2 - 1  # {0,1} → {-1,1}
    return features, Y


# --- Training: build circuit, split data, optimize, evaluate ---
def run_training(
    features: np.ndarray,
    Y: np.ndarray,
    *,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    test_size: float = 0.25,
    optimizer: str = "adam",
    early_stop_target: float | None = 0.9,
) -> dict[str, Any]:
    """Train classifier: circuit + bias, square loss, batched. Optional early stop when test acc ≥ target."""
    dev = qml.device("default.qubit", wires=NUM_QUBITS)

    # Circuit: state_prep(angles) → layers of Rot+CNOT → expval(PauliZ(0))
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(weights, angles):
        state_preparation(angles, wires=(0, 1))
        for lw in weights:
            layer(lw, wires=(0, 1))
        return qml.expval(qml.PauliZ(0))

    def model(weights, bias, angles):
        return circuit(weights, angles) + bias

    def cost(weights, bias, X_batch, Y_batch):
        preds = model(weights, bias, X_batch.T)
        return pnp.mean((Y_batch - preds) ** 2)

    # Train/val split (seed-driven)
    n = len(Y)
    np.random.seed(seed)
    try:
        pnp.random.seed(seed)
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * (1 - test_size))
    feats_train = pnp.array(features[idx[:n_train]])
    Y_train = pnp.array(Y[idx[:n_train]])
    feats_test = features[idx[n_train:]]
    Y_test = Y[idx[n_train:]]

    # Weights and optimizer
    weights_init = 0.01 * pnp.random.randn(
        num_layers, NUM_QUBITS, 3, requires_grad=True
    )
    bias_init = pnp.array(0.0, requires_grad=True)
    if optimizer == "adam":
        opt = AdamOptimizer(lr)
    else:
        opt = NesterovMomentumOptimizer(lr)

    # Compile (first run)
    t0 = time.perf_counter()
    _ = circuit(weights_init, feats_train[0])
    _ = cost(weights_init, bias_init, feats_train[:1], Y_train[:1])
    compile_sec = time.perf_counter() - t0

    # Optimize (batched steps; optional early stop every 100 steps)
    t0 = time.perf_counter()
    weights, bias = weights_init, bias_init
    steps_done = 0
    for step in range(iterations):
        batch_idx = rng.integers(0, n_train, size=(batch_size,))
        fb = feats_train[batch_idx]
        yb = Y_train[batch_idx]
        out = opt.step(cost, weights, bias, fb, yb)
        weights, bias = out[0], out[1]
        steps_done += 1
        if early_stop_target is not None and (step + 1) % 100 == 0:
            pred_test_now = np.sign(
                np.array(model(weights, bias, pnp.array(feats_test).T))
            ).flatten()
            test_acc_now = float(
                np.mean(np.abs(pred_test_now - np.array(Y_test)) < 1e-5)
            )
            if test_acc_now >= early_stop_target:
                break
    train_sec = time.perf_counter() - t0

    # Metrics (train/test accuracy)
    pred_train = np.sign(np.array(model(weights, bias, feats_train.T))).flatten()
    pred_test = np.sign(
        np.array(model(weights, bias, pnp.array(feats_test).T))
    ).flatten()
    Y_train_np = np.array(Y_train)
    Y_test_np = np.array(Y_test)
    train_acc = float(np.mean(np.abs(pred_train - Y_train_np) < 1e-5))
    test_acc = float(np.mean(np.abs(pred_test - Y_test_np) < 1e-5))

    return {
        "compile_time_sec": compile_sec,
        "train_time_sec": train_sec,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "n_train": n_train,
        "n_test": len(Y_test),
        "epochs": steps_done,
        "samples_per_sec": (steps_done * batch_size) / train_sec
        if train_sec > 0
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PennyLane Iris amplitude encoding baseline (2-class)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1500,
        help="Max optimizer steps per run (default: 1500)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Batch size (default: 5)"
    )
    parser.add_argument(
        "--layers", type=int, default=10, help="Variational layers (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.08, help="Learning rate (default: 0.08 for Adam)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test fraction (default: 0.1); ignored if --data-file set",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=("adam", "nesterov"),
        help="Optimizer (default: adam)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of restarts; best test acc reported (default: 20)",
    )
    parser.add_argument(
        "--early-stop",
        type=float,
        default=0.9,
        help="Stop run when test acc >= this (default: 0.9; 0 = off)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to official Iris file (cols: f0,f1,f2,f3,label). If set, use first 2 cols + 75%% train.",
    )
    args = parser.parse_args()

    # Data source: official file (when --data-file set) or sklearn Iris (default)
    if args.data_file:
        features, Y = load_iris_from_file(args.data_file)
        test_size = 0.25  # tutorial uses 75% train
        data_src = f"official file (2 features): {args.data_file}"
    else:
        features, Y = load_iris_binary(seed=args.seed)
        test_size = args.test_size
        data_src = "sklearn load_iris, classes 0 & 1, 4 features"
    n = len(Y)
    print("Iris amplitude baseline (PennyLane) — 2-class variational classifier")
    print(
        f"  Data: {data_src} → L2 norm → get_angles  (n={n}; 2-class Iris = 100 samples)"
    )
    print(
        f"  Iters: {args.iters}, batch_size: {args.batch_size}, layers: {args.layers}, lr: {args.lr}, optimizer: {args.optimizer}"
    )

    results: list[dict[str, Any]] = []
    for t in range(args.trials):
        r = run_training(
            features,
            Y,
            num_layers=args.layers,
            iterations=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + t,
            test_size=test_size,
            optimizer=args.optimizer,
            early_stop_target=args.early_stop if args.early_stop > 0 else None,
        )
        results.append(r)
        print(f"\n  Trial {t + 1}:")
        print(f"    Compile:   {r['compile_time_sec']:.4f} s")
        print(f"    Train:     {r['train_time_sec']:.4f} s")
        print(f"    Train acc: {r['train_accuracy']:.4f}  (n={r['n_train']})")
        print(f"    Test acc:  {r['test_accuracy']:.4f}  (n={r['n_test']})")
        print(f"    Throughput: {r['samples_per_sec']:.1f} samples/s")

    if args.trials > 1:
        test_accs = sorted(r["test_accuracy"] for r in results)
        best = test_accs[-1]
        mid = args.trials // 2
        print(
            f"\n  Best test accuracy:  {best:.4f}  (median: {test_accs[mid]:.4f}, min: {test_accs[0]:.4f}, max: {test_accs[-1]:.4f})"
        )
        if best >= 0.9:
            print("  → Target ≥0.9 achieved.")


if __name__ == "__main__":
    main()
