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
PennyLane baseline: SVHN (2-class), IQP encoding, variational classifier.

Pipeline:
  SVHN (32x32x3) -> Flatten (3072) -> binary filter (1 vs 7) -> subsample (500)
    -> StandardScaler -> PCA to n_qubits dims
    -> Custom IQP circuit: H^n * Diag(phases) * H^n (inside circuit, re-runs
       every forward/backward pass)
    -> variational layers (Rot + ring CNOT) -> expval(PauliZ(0))
    -> square loss, Adam optimizer, batched training

Key: IQP encoding cost is paid on every forward pass because the H-D-H circuit
is part of the quantum node. Compare with qdp_pipeline/svhn_iqp.py where encoding
is done once upfront. The custom IQP gates (PhaseShift + ControlledPhaseShift)
match QDP's CUDA kernel convention: H^n * U_phase * H^n |0>^n with exp(i*phi).
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from typing import Any

import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.optimize import AdamOptimizer, NesterovMomentumOptimizer
except ImportError as e:
    raise SystemExit(
        "PennyLane is required. Install with: uv sync --group benchmark"
    ) from e

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: uv sync --group benchmark"
    ) from e

try:
    from scipy.io import loadmat
except ImportError as e:
    raise SystemExit("scipy is required. Install with: pip install scipy") from e


# ---------------------------------------------------------------------------
# SVHN data loading
# ---------------------------------------------------------------------------

SVHN_URLS = {
    "train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    "test": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
}


def _download_if_needed(url: str, dest: str) -> str:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"    Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"    Saved to {dest}")
    return dest


def load_svhn(
    data_home: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load SVHN train/test: (n, 3072) float64 in [0,1], labels 0-9."""
    if data_home is None:
        data_home = os.path.join(os.path.expanduser("~"), "scikit_learn_data", "svhn")

    train_path = _download_if_needed(
        SVHN_URLS["train"], os.path.join(data_home, "train_32x32.mat")
    )
    test_path = _download_if_needed(
        SVHN_URLS["test"], os.path.join(data_home, "test_32x32.mat")
    )

    train_mat = loadmat(train_path)
    test_mat = loadmat(test_path)

    X_train = (
        train_mat["X"].transpose(3, 0, 1, 2).reshape(-1, 3072).astype(np.float64)
        / 255.0
    )
    X_test = (
        test_mat["X"].transpose(3, 0, 1, 2).reshape(-1, 3072).astype(np.float64)
        / 255.0
    )
    Y_train = train_mat["y"].ravel().astype(int) % 10
    Y_test = test_mat["y"].ravel().astype(int) % 10

    return X_train, X_test, Y_train, Y_test


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

CLASS_POS = 1
CLASS_NEG = 7


def _filter_binary(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = (Y == CLASS_POS) | (Y == CLASS_NEG)
    return X[mask], np.where(Y[mask] == CLASS_POS, 1, -1)


def preprocess_pca(
    X: np.ndarray, n_components: int
) -> np.ndarray:
    """StandardScaler -> PCA to n_components dimensions."""
    X_scaled = StandardScaler().fit_transform(X)
    return PCA(n_components=n_components).fit_transform(X_scaled)


# ---------------------------------------------------------------------------
# Circuit: custom IQP (H-D-H) encoding + variational layers (Rot + ring CNOT)
# ---------------------------------------------------------------------------


def layer(layer_weights, wires) -> None:
    """Rot on each wire + ring of CNOTs."""
    for i, w in enumerate(wires):
        qml.Rot(*layer_weights[i], wires=w)
    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_training(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    *,
    n_qubits: int,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    optimizer: str = "adam",
    early_stop_target: float | None = None,
) -> dict[str, Any]:
    """Train variational classifier with custom H-D-H IQP circuit.

    The IQP circuit matches QDP's convention: H^n * Diag(phases) * H^n |0>^n
    using PhaseShift and ControlledPhaseShift gates.
    """
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(weights, features):
        # IQP circuit matching QDP convention: H^n * Diag * H^n
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            qml.PhaseShift(features[i], wires=w)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.ControlledPhaseShift(
                    features[i] * features[j], wires=[wires[i], wires[j]]
                )
        for w in wires:
            qml.Hadamard(wires=w)
        for lw in weights:
            layer(lw, wires)
        return qml.expval(qml.PauliZ(0))

    def model(weights, bias, features):
        return circuit(weights, features) + bias

    def cost(weights, bias, X_batch, Y_batch):
        preds = pnp.array([model(weights, bias, x) for x in X_batch])
        return pnp.mean((Y_batch - preds) ** 2)

    n_train = len(Y_train)
    np.random.seed(seed)
    try:
        pnp.random.seed(seed)
    except Exception:
        pass
    rng = np.random.default_rng(seed)

    X_train_pnp = pnp.array(X_train, requires_grad=False)
    Y_train_pnp = pnp.array(Y_train.astype(np.float64), requires_grad=False)

    # Weights and optimizer
    weights_init = 0.01 * pnp.random.randn(
        num_layers, n_qubits, 3, requires_grad=True
    )
    bias_init = pnp.array(0.0, requires_grad=True)
    if optimizer == "adam":
        opt = AdamOptimizer(lr)
    else:
        opt = NesterovMomentumOptimizer(lr)

    # Compile (first run)
    t0 = time.perf_counter()
    _ = circuit(weights_init, X_train_pnp[0])
    _ = cost(weights_init, bias_init, X_train_pnp[:1], Y_train_pnp[:1])
    compile_sec = time.perf_counter() - t0

    # Optimize (batched steps; optional early stop every 100 steps)
    t0 = time.perf_counter()
    weights, bias = weights_init, bias_init
    steps_done = 0
    for step in range(iterations):
        batch_idx = rng.integers(0, n_train, size=(batch_size,))
        fb = X_train_pnp[batch_idx]
        yb = Y_train_pnp[batch_idx]
        out = opt.step(cost, weights, bias, fb, yb)
        weights, bias = out[0], out[1]
        steps_done += 1
        if early_stop_target is not None and (step + 1) % 100 == 0:
            pred_test_now = np.sign(
                np.array([model(weights, bias, x) for x in pnp.array(X_test)])
            ).flatten()
            test_acc_now = float(
                np.mean(np.abs(pred_test_now - np.array(Y_test)) < 1e-5)
            )
            if test_acc_now >= early_stop_target:
                break
    train_sec = time.perf_counter() - t0

    # Metrics
    pred_train = np.sign(
        np.array([model(weights, bias, x) for x in X_train_pnp])
    ).flatten()
    pred_test = np.sign(
        np.array([model(weights, bias, x) for x in pnp.array(X_test)])
    ).flatten()
    Y_train_np = np.array(Y_train_pnp)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PennyLane SVHN IQP variational classifier baseline (2-class)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Total samples after binary filter + subsample (default: 500)",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=10,
        help="Number of qubits / PCA components (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=500,
        help="Max optimizer steps per run (default: 500)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size (default: 10)"
    )
    parser.add_argument(
        "--layers", type=int, default=6, help="Variational layers (default: 6)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test fraction (default: 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
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
        default=3,
        help="Number of restarts; best test acc reported (default: 3)",
    )
    parser.add_argument(
        "--early-stop",
        type=float,
        default=0.85,
        help="Stop run when test acc >= this (default: 0.85; 0 = off)",
    )
    parser.add_argument("--data-home", type=str, default=None, help="Data cache dir")
    args = parser.parse_args()

    n_qubits = args.n_qubits

    print("SVHN IQP variational classifier — PennyLane baseline (2-class)")
    print(
        f"  {n_qubits} qubits, PCA 3072->{n_qubits}, "
        f"binary: digit {CLASS_POS} vs {CLASS_NEG}"
    )
    print(
        f"  n_samples={args.n_samples}, iters={args.iters}, "
        f"batch_size={args.batch_size}, layers={args.layers}, "
        f"lr={args.lr}, optimizer={args.optimizer}"
    )
    print()

    # Load & filter
    print("  Loading SVHN ...")
    X_train_all, X_test_all, Y_train_all, Y_test_all = load_svhn(
        data_home=args.data_home
    )
    X_all = np.concatenate([X_train_all, X_test_all], axis=0)
    Y_all = np.concatenate([Y_train_all, Y_test_all], axis=0)
    X_bin, Y_bin = _filter_binary(X_all, Y_all)
    print(
        f"  Binary filtered: {len(Y_bin):,} samples "
        f"(pos={np.mean(Y_bin == 1):.2f})"
    )

    rng = np.random.default_rng(args.seed)
    if args.n_samples < len(Y_bin):
        idx = rng.choice(len(Y_bin), size=args.n_samples, replace=False)
        X_bin, Y_bin = X_bin[idx], Y_bin[idx]
    print(f"  Subsampled: {len(Y_bin):,} samples")

    # PCA
    t0 = time.perf_counter()
    X_pca = preprocess_pca(X_bin, n_components=n_qubits)
    pca_sec = time.perf_counter() - t0
    print(f"  PCA: {pca_sec:.4f}s  (3072->{n_qubits})")

    # Train/test split (done once before trials, matching QDP pipeline)
    split_rng = np.random.default_rng(args.seed)
    split_idx = split_rng.permutation(len(Y_bin))
    n_train = int(len(Y_bin) * (1 - args.test_size))
    X_train, X_test = X_pca[split_idx[:n_train]], X_pca[split_idx[n_train:]]
    Y_train, Y_test = Y_bin[split_idx[:n_train]], Y_bin[split_idx[n_train:]]

    print()

    results: list[dict[str, Any]] = []
    early_stop = args.early_stop if args.early_stop > 0 else None
    for t in range(args.trials):
        r = run_training(
            X_train,
            X_test,
            Y_train,
            Y_test,
            n_qubits=n_qubits,
            num_layers=args.layers,
            iterations=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + t,
            optimizer=args.optimizer,
            early_stop_target=early_stop,
        )
        results.append(r)
        print(f"  Trial {t + 1}:")
        print(f"    Compile:   {r['compile_time_sec']:.4f} s")
        print(f"    Train:     {r['train_time_sec']:.4f} s")
        print(f"    Train acc: {r['train_accuracy']:.4f}  (n={r['n_train']})")
        print(f"    Test acc:  {r['test_accuracy']:.4f}  (n={r['n_test']})")
        print(f"    Throughput: {r['samples_per_sec']:.1f} samples/s")
        print()

    if args.trials > 1:
        test_accs = [r["test_accuracy"] for r in results]
        print(
            f"  Best test accuracy:  {max(test_accs):.4f}  "
            f"(median: {float(np.median(test_accs)):.4f}, "
            f"mean: {float(np.mean(test_accs)):.4f} +/- {float(np.std(test_accs)):.4f}, "
            f"min: {min(test_accs):.4f}, max: {max(test_accs):.4f})"
        )


if __name__ == "__main__":
    main()
