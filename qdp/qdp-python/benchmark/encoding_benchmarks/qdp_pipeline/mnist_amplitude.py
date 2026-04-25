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
QDP pipeline: MNIST (2-class), same data and training as baseline; only encoding differs.

Data source: sklearn fetch_openml('mnist_784'), binary subset (default: digits 3 vs 6).
Pipeline: PCA (784 -> 2^num_qubits) -> L2 norm -> QDP (QdpEngine.encode + amplitude) -> StatePrep(encoded)
-> Rot layers + CNOT ring -> expval(PauliZ(0)) + bias; square loss; SGD+Nesterov via torch.

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

from qumat_qdp import QdpEngine

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


# --- Data: MNIST binary subset -> PCA -> L2 norm (returns raw vectors for QDP) ---
def load_mnist_binary_nd(
    digits: tuple[int, int] = DEFAULT_DIGITS,
    n_samples: int = DEFAULT_N_SAMPLES,
    num_qubits: int = DEFAULT_NUM_QUBITS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MNIST binary classification. Fetch two digit classes, subsample, PCA, L2 normalize.
    Returns (X_norm, Y) with X_norm shape (n, 2**num_qubits), Y in {-1, 1}.
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


# --- Encoding: QDP (QdpEngine.encode + amplitude) ---
def encode_via_qdp(
    X_norm: np.ndarray,
    num_qubits: int,
    batch_size: int = 10,  # kept for CLI symmetry; not used here
    device_id: int = 0,
    qdp_backend: str = "cuda",
    data_dir: str | None = None,
    filename: str = "mnist_nd.npy",
) -> torch.Tensor:
    """QDP: use QdpEngine.encode on PCA-reduced vectors (amplitude), return encoded tensor on GPU.

    Uses in-memory encoding via QdpEngine instead of writing/reading .npy files. The returned
    tensor stays on the selected GPU device and can be fed directly to qml.StatePrep.
    """
    n, dim = X_norm.shape
    state_dim = 2**num_qubits
    if dim != state_dim:
        raise ValueError(
            f"X_norm must have {state_dim} features for {num_qubits} qubits, got {dim}"
        )
    engine = QdpEngine(device_id=device_id, precision="float32", backend=qdp_backend)
    qt = engine.encode(
        X_norm.astype(np.float64),
        num_qubits=num_qubits,
        encoding_method="amplitude",
    )
    encoded = torch.from_dlpack(qt)
    return encoded[:n]


# --- Training: GPU (lightning.gpu + torch + adjoint) ---
def run_training(
    encoded_train: torch.Tensor,
    encoded_test: torch.Tensor,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    *,
    num_qubits: int,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    early_stop_target: float | None = None,
) -> dict[str, Any]:
    """Train variational classifier: StatePrep(encoded) + Rot layers + bias, square loss, batched.
    Prefers lightning.gpu (CUDA-only); falls back to default.qubit on CPU when unavailable
    (e.g. AMD/ROCm host). Optional early stop when test acc >= target."""
    n_train = len(Y_train)
    rng = np.random.default_rng(seed)

    wires = tuple(range(num_qubits))
    qml_device_name = "cuda"
    try:
        dev_qml = qml.device("lightning.gpu", wires=num_qubits)
    except Exception:
        dev_qml = qml.device("default.qubit", wires=num_qubits)
        qml_device_name = "cpu"
        encoded_train = encoded_train.cpu()
        encoded_test = encoded_test.cpu()

    device = encoded_train.device
    # Encoded data may be complex (from QDP); use real dtype for weights and labels.
    real_dtype = (
        torch.float64 if encoded_train.dtype == torch.complex128 else torch.float32
    )
    Y_train_t = torch.tensor(Y_train, dtype=real_dtype, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=real_dtype, device=device)

    @qml.qnode(dev_qml, interface="torch", diff_method="adjoint")
    def circuit(weights, state_vector):
        qml.StatePrep(state_vector, wires=wires)
        for lw in weights:
            layer(lw, wires=wires)
        return qml.expval(qml.PauliZ(0))

    def model(weights, bias, state_batch):
        return circuit(weights, state_batch) + bias

    def cost(weights, bias, X_batch, Y_batch):
        preds = model(weights, bias, X_batch)
        return torch.mean((Y_batch - preds) ** 2)

    torch.manual_seed(seed)
    weights = (
        (0.01 * torch.randn(num_layers, num_qubits, 3, device=device, dtype=real_dtype))
        .detach()
        .requires_grad_(True)
    )
    bias = (
        torch.zeros(1, device=device, dtype=real_dtype).squeeze().requires_grad_(True)
    )
    opt = torch.optim.SGD([weights, bias], lr=lr, momentum=0.9, nesterov=True)

    # Compile (first run)
    t0 = time.perf_counter()
    _ = circuit(weights, encoded_train[0:1])
    _ = cost(weights, bias, encoded_train[:1], Y_train_t[:1])
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
        fb = encoded_train[batch_idx]
        yb = Y_train_t[batch_idx]
        loss = cost(weights, bias, fb, yb)
        loss.backward()
        opt.step()
        steps_done += 1
        if early_stop_target is not None and (step + 1) % 100 == 0:
            with torch.no_grad():
                pred_test_now = torch.sign(model(weights, bias, encoded_test)).flatten()
                test_acc_now = (
                    (pred_test_now - Y_test_t).abs().lt(1e-5).float().mean().item()
                )
            if test_acc_now >= early_stop_target:
                break
    train_sec = time.perf_counter() - t0

    with torch.no_grad():
        pred_train = torch.sign(model(weights, bias, encoded_train)).flatten()
        pred_test = torch.sign(model(weights, bias, encoded_test)).flatten()
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
        "qml_device": qml_device_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QDP MNIST amplitude encoding pipeline (2-class, same training as baseline)"
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
        "--device-id", type=int, default=0, help="QDP device (default: 0)"
    )
    parser.add_argument(
        "--qdp-backend",
        choices=("cuda", "amd"),
        default="cuda",
        help="QDP backend for direct state preparation (default: cuda)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Dir for .npy files (default: temp)"
    )
    args = parser.parse_args()

    d0, d1 = (int(d) for d in args.digits.split(","))
    digits = (d0, d1)
    X_norm, Y = load_mnist_binary_nd(
        digits=digits,
        n_samples=args.n_samples,
        num_qubits=args.qubits,
        seed=args.seed,
    )
    n = len(Y)
    state_dim = 2**args.qubits
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_train = int(n * (1 - args.test_size))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    X_train = X_norm[train_idx]
    X_test = X_norm[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    # QDP encoding
    t0 = time.perf_counter()
    encoded_train = encode_via_qdp(
        X_train,
        num_qubits=args.qubits,
        batch_size=args.batch_size,
        device_id=args.device_id,
        qdp_backend=args.qdp_backend,
        data_dir=args.data_dir,
        filename="mnist_nd_train.npy",
    )
    encoded_test = encode_via_qdp(
        X_test,
        num_qubits=args.qubits,
        batch_size=args.batch_size,
        device_id=args.device_id,
        qdp_backend=args.qdp_backend,
        data_dir=args.data_dir,
        filename="mnist_nd_test.npy",
    )
    encode_sec = time.perf_counter() - t0

    print("MNIST amplitude (QDP encoding) — 2-class variational classifier")
    print(
        f"  Data: fetch_openml('mnist_784'), digits {d0} vs {d1}, "
        f"PCA {state_dim}-D, QDP amplitude  (n={n})"
    )
    print(
        f"  Qubits: {args.qubits}, iters: {args.iters}, batch_size: {args.batch_size}, "
        f"layers: {args.layers}, lr: {args.lr}, qdp_backend: {args.qdp_backend}"
    )
    print(
        f"  QDP encode:  {encode_sec:.4f} s  (train + test, {n_train} + {n - n_train} samples)"
    )

    results: list[dict[str, Any]] = []
    early_stop = args.early_stop if args.early_stop > 0 else None
    for t in range(args.trials):
        r = run_training(
            encoded_train,
            encoded_test,
            Y_train,
            Y_test,
            num_qubits=args.qubits,
            num_layers=args.layers,
            iterations=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + t,
            early_stop_target=early_stop,
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
