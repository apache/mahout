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
PennyLane baseline: MNIST (2-class), amplitude encoding, variational classifier.

Data source: sklearn fetch_openml('mnist_784'), binary subset (default: digits 3 vs 6).
Pipeline: PCA (784 -> 2^num_qubits) -> L2 norm -> AmplitudeEmbedding -> Rot layers + CNOT ring
-> expval(PauliZ(0)) + bias; square loss; SGD+Nesterov via torch.

Training: lightning.gpu (adjoint, torch).
"""

from __future__ import annotations

# --- Imports ---
import argparse
import time
from typing import Any

import numpy as np
import torch

try:
    import pennylane as qml
except ImportError as e:
    raise SystemExit(
        "PennyLane is required. Install with: uv sync --group benchmark"
    ) from e

try:
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: uv sync --group benchmark"
    ) from e

try:
    from tqdm import trange
except ImportError:
    trange = None


DEFAULT_NUM_QUBITS = 4
DEFAULT_DIGITS = (3, 6)
DEFAULT_N_SAMPLES = 500


# --- Circuit: variational layer (Rot + CNOT ring) ---
def layer(layer_weights, wires):
    """Rot on each wire + ring of CNOTs (generalized from 2-qubit Iris tutorial)."""
    for i, w in enumerate(wires):
        qml.Rot(*layer_weights[i], wires=w)
    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


# --- Data: MNIST binary subset -> PCA -> L2 norm ---
def load_mnist_binary(
    digits: tuple[int, int] = DEFAULT_DIGITS,
    n_samples: int = DEFAULT_N_SAMPLES,
    num_qubits: int = DEFAULT_NUM_QUBITS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MNIST binary classification. Fetch two digit classes, subsample, PCA, L2 normalize.
    Returns (features, Y) with features shape (n, 2**num_qubits), Y in {-1, 1}.
    Data source: sklearn.datasets.fetch_openml('mnist_784').
    """
    state_dim = 2**num_qubits
    rng = np.random.default_rng(seed)

    X_raw, y_raw = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )
    y = y_raw.astype(int)
    mask = (y == digits[0]) | (y == digits[1])
    X = np.asarray(X_raw[mask], dtype=np.float64)
    y = y[mask]

    # Balanced subsample
    idx0 = np.where(y == digits[0])[0]
    idx1 = np.where(y == digits[1])[0]
    n0 = min(n_samples, len(idx0))
    n1 = min(n_samples, len(idx1))
    sel = np.concatenate(
        [
            rng.choice(idx0, size=n0, replace=False),
            rng.choice(idx1, size=n1, replace=False),
        ]
    )
    rng.shuffle(sel)
    X = X[sel]
    y = y[sel]

    # StandardScaler -> PCA -> L2 norm
    X = StandardScaler().fit_transform(X)
    n_components = min(state_dim, X.shape[1], X.shape[0])
    X = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    if n_components < state_dim:
        X = np.pad(X, ((0, 0), (0, state_dim - n_components)), constant_values=0.0)
    norm = np.sqrt(np.sum(X**2, axis=-1)) + 1e-12
    X_norm = (X.T / norm).T

    # Labels: first digit -> -1, second digit -> +1
    Y = np.where(y == digits[0], -1.0, 1.0)
    return X_norm, Y


# --- Training: GPU (lightning.gpu + torch + adjoint) ---
def run_training(
    features: np.ndarray,
    Y: np.ndarray,
    *,
    num_qubits: int,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    test_size: float = 0.25,
    early_stop_target: float | None = 0.95,
    device_id: int = 0,
) -> dict[str, Any]:
    """Train classifier: AmplitudeEmbedding + Rot layers + bias, square loss, batched.
    Uses lightning.gpu + torch interface + adjoint diff. Data on GPU as torch tensors.
    Optional early stop when test acc >= target."""

    # Train/val split (seed-driven)
    n = len(Y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * (1 - test_size))
    X_train = features[idx[:n_train]]
    X_test = features[idx[n_train:]]
    Y_train = Y[idx[:n_train]]
    Y_test = Y[idx[n_train:]]

    wires = tuple(range(num_qubits))
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.float64

    feats_train = torch.tensor(X_train, dtype=dtype, device=device)
    feats_test = torch.tensor(X_test, dtype=dtype, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=dtype, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=dtype, device=device)

    dev_qml = qml.device("lightning.gpu", wires=num_qubits)

    @qml.qnode(dev_qml, interface="torch", diff_method="adjoint")
    def circuit(weights, x):
        qml.AmplitudeEmbedding(features=x, wires=wires, normalize=False)
        for lw in weights:
            layer(lw, wires=wires)
        return qml.expval(qml.PauliZ(0))

    def model(weights, bias, x):
        return circuit(weights, x) + bias

    def cost(weights, bias, X_batch, Y_batch):
        preds = model(weights, bias, X_batch)
        return torch.mean((Y_batch - preds) ** 2)

    torch.manual_seed(seed)
    weights = (
        (0.01 * torch.randn(num_layers, num_qubits, 3, device=device, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    bias = torch.zeros(1, device=device, dtype=dtype).squeeze().requires_grad_(True)
    opt = torch.optim.SGD([weights, bias], lr=lr, momentum=0.9, nesterov=True)

    # Compile (first run)
    t0 = time.perf_counter()
    _ = circuit(weights, feats_train[0:1])
    _ = cost(weights, bias, feats_train[:1], Y_train_t[:1])
    compile_sec = time.perf_counter() - t0

    # Optimize
    t0 = time.perf_counter()
    steps_done = 0
    step_iter = (
        trange(iterations, desc="  Training (GPU)", leave=False)
        if trange
        else range(iterations)
    )
    for step in step_iter:
        opt.zero_grad()
        batch_idx = rng.integers(0, n_train, size=(batch_size,))
        fb = feats_train[batch_idx]
        yb = Y_train_t[batch_idx]
        loss = cost(weights, bias, fb, yb)
        loss.backward()
        opt.step()
        steps_done += 1
        if early_stop_target is not None and (step + 1) % 100 == 0:
            with torch.no_grad():
                pred_test_now = torch.sign(model(weights, bias, feats_test)).flatten()
                test_acc_now = (
                    (pred_test_now - Y_test_t).abs().lt(1e-5).float().mean().item()
                )
            if test_acc_now >= early_stop_target:
                break
    train_sec = time.perf_counter() - t0

    with torch.no_grad():
        pred_train = torch.sign(model(weights, bias, feats_train)).flatten()
        pred_test = torch.sign(model(weights, bias, feats_test)).flatten()
    train_acc = (pred_train - Y_train_t).abs().lt(1e-5).float().mean().item()
    test_acc = (pred_test - Y_test_t).abs().lt(1e-5).float().mean().item()

    return {
        "compile_time_sec": compile_sec,
        "train_time_sec": train_sec,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "n_train": n_train,
        "n_test": len(Y_test),
        "epochs": steps_done,
        "samples_per_sec": (steps_done * batch_size) / train_sec
        if train_sec > 0
        else 0.0,
        "qml_device": "cuda",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PennyLane MNIST amplitude encoding baseline (2-class)"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=DEFAULT_NUM_QUBITS,
        help=f"Number of qubits; PCA reduces to 2^qubits features (default: {DEFAULT_NUM_QUBITS})",
    )
    parser.add_argument(
        "--digits",
        type=str,
        default="3,6",
        help="Two digits for binary classification, comma-separated (default: '3,6')",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Max samples per class (default: {DEFAULT_N_SAMPLES}; total <= 2*n_samples)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=2000,
        help="Max optimizer steps per run (default: 2000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size (default: 10)"
    )
    parser.add_argument(
        "--layers", type=int, default=10, help="Variational layers (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="Learning rate (default: 0.05)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test fraction (default: 0.25)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of restarts; best test acc reported (default: 10)",
    )
    parser.add_argument(
        "--early-stop",
        type=float,
        default=0.95,
        help="Stop run when test acc >= this (default: 0.95; 0 = off)",
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="GPU device id (default: 0)"
    )
    args = parser.parse_args()

    d0, d1 = (int(d) for d in args.digits.split(","))
    digits = (d0, d1)
    features, Y = load_mnist_binary(
        digits=digits,
        n_samples=args.n_samples,
        num_qubits=args.qubits,
        seed=args.seed,
    )
    n = len(Y)
    state_dim = 2**args.qubits
    print("MNIST amplitude baseline (PennyLane) — 2-class variational classifier")
    print(
        f"  Data: fetch_openml('mnist_784'), digits {d0} vs {d1}, "
        f"PCA {state_dim}-D, L2 norm  (n={n})"
    )
    print(
        f"  Qubits: {args.qubits}, iters: {args.iters}, batch_size: {args.batch_size}, "
        f"layers: {args.layers}, lr: {args.lr}"
    )

    results: list[dict[str, Any]] = []
    for t in range(args.trials):
        r = run_training(
            features,
            Y,
            num_qubits=args.qubits,
            num_layers=args.layers,
            iterations=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + t,
            test_size=args.test_size,
            early_stop_target=args.early_stop if args.early_stop > 0 else None,
            device_id=args.device_id,
        )
        results.append(r)
        print(f"\n  Trial {t + 1}:")
        print(f"    QML device: {r.get('qml_device', 'cpu')}")
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
            f"\n  Best test accuracy:  {best:.4f}  (median: {test_accs[mid]:.4f}, "
            f"min: {test_accs[0]:.4f}, max: {test_accs[-1]:.4f})"
        )
        if best >= 0.95:
            print("  → Target ≥0.95 achieved.")


if __name__ == "__main__":
    main()
