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
QDP pipeline: SVHN (2-class), IQP encoding done once upfront, variational classifier.

Pipeline:
  SVHN (32x32x3) -> Flatten (3072) -> binary filter (1 vs 7) -> subsample (500)
    -> StandardScaler -> PCA to n_qubits dims
    -> features_to_iqp_params -> QdpEngine.encode(method="iqp") (one-time, GPU)
    -> StatePrep(encoded) -> variational layers (Rot + ring CNOT) -> expval(PauliZ(0))
    -> square loss, optimizer, batched training

Backends:
  --backend cpu  : default.qubit + autograd + backprop + PennyLane optimizer
                   (encoded tensors copied GPU -> CPU via .cpu().numpy())
  --backend gpu  : lightning.gpu + torch + adjoint + PyTorch optimizer
                   (zero copy: encoded tensors stay on GPU)

Key: IQP encoding is done once upfront via QDP GPU kernels. The training circuit
uses StatePrep to load pre-encoded states, so encoding cost is NOT paid per step.
Compare with pennylane_baseline/svhn_iqp.py where IQPEmbedding runs every forward pass.
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from typing import Any

import numpy as np
import torch

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
    raise SystemExit(
        "scipy is required. Install with: uv sync --group benchmark"
    ) from e

try:
    from qumat_qdp import QdpEngine
except ImportError as e:
    raise SystemExit(
        "qumat_qdp is required. Install with: uv sync --group benchmark"
    ) from e


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
        test_mat["X"].transpose(3, 0, 1, 2).reshape(-1, 3072).astype(np.float64) / 255.0
    )
    Y_train = train_mat["y"].ravel().astype(int) % 10
    Y_test = test_mat["y"].ravel().astype(int) % 10

    return X_train, X_test, Y_train, Y_test


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

CLASS_POS = 1
CLASS_NEG = 7


def _filter_binary(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (Y == CLASS_POS) | (Y == CLASS_NEG)
    return X[mask], np.where(Y[mask] == CLASS_POS, 1, -1)


def preprocess_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """StandardScaler -> PCA to n_components dimensions."""
    X_scaled = StandardScaler().fit_transform(X)
    return PCA(n_components=n_components).fit_transform(X_scaled)


# ---------------------------------------------------------------------------
# IQP encoding via QDP
# ---------------------------------------------------------------------------


def features_to_iqp_params(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert PCA features to QDP IQP parameter vectors.

    QDP IQP expects n + n*(n-1)/2 parameters per sample:
      [z_0, ..., z_{n-1}, zz_{0,1}, zz_{0,2}, ..., zz_{n-2,n-1}]

    Where z_i = features[i] and zz_{i,j} = features[i] * features[j].
    """
    n_pairs = n_qubits * (n_qubits - 1) // 2
    params = np.empty((X.shape[0], n_qubits + n_pairs), dtype=np.float64)

    # Single-qubit Z angles
    params[:, :n_qubits] = X

    # Two-qubit ZZ angles: features[i] * features[j] for i < j
    pair_idx = n_qubits
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            params[:, pair_idx] = X[:, i] * X[:, j]
            pair_idx += 1

    return params


def encode_qdp(X_params: np.ndarray, n_qubits: int, device_id: int = 0) -> torch.Tensor:
    """QdpEngine IQP batch encode -> CUDA complex tensor."""
    engine = QdpEngine(device_id=device_id, precision="float64")
    qt = engine.encode(
        X_params.astype(np.float64),
        num_qubits=n_qubits,
        encoding_method="iqp",
    )
    encoded = torch.from_dlpack(qt)
    return encoded[: X_params.shape[0]]


# ---------------------------------------------------------------------------
# Circuit: variational layers (Rot + ring CNOT)
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
    encoded_train: np.ndarray | torch.Tensor,
    encoded_test: np.ndarray | torch.Tensor,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    *,
    n_qubits: int,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    early_stop_target: float | None = None,
    optimizer: str = "adam",
    backend: str = "cpu",
) -> dict[str, Any]:
    """Train variational classifier with StatePrep(encoded) -- no encoding per step."""
    if backend == "gpu":
        return _run_training_gpu(
            encoded_train,
            encoded_test,
            Y_train,
            Y_test,
            n_qubits=n_qubits,
            num_layers=num_layers,
            iterations=iterations,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            early_stop_target=early_stop_target,
            optimizer=optimizer,
        )
    return _run_training_cpu(
        encoded_train,
        encoded_test,
        Y_train,
        Y_test,
        n_qubits=n_qubits,
        num_layers=num_layers,
        iterations=iterations,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        early_stop_target=early_stop_target,
        optimizer=optimizer,
    )


def _run_training_cpu(
    encoded_train: np.ndarray,
    encoded_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    *,
    n_qubits: int,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    early_stop_target: float | None = None,
    optimizer: str = "adam",
) -> dict[str, Any]:
    """CPU path: default.qubit + autograd + backprop + PennyLane optimizer."""
    wires = list(range(n_qubits))
    n_train = len(Y_train)
    np.random.seed(seed)
    try:
        pnp.random.seed(seed)
    except Exception:
        pass
    rng = np.random.default_rng(seed)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(weights, state_vector):
        qml.StatePrep(state_vector, wires=wires)
        for lw in weights:
            layer(lw, wires)
        return qml.expval(qml.PauliZ(0))

    def model(weights, bias, state_vector):
        return circuit(weights, state_vector) + bias

    def cost(weights, bias, X_batch, Y_batch):
        preds = pnp.array([model(weights, bias, x) for x in X_batch])
        return pnp.mean((Y_batch - preds) ** 2)

    feats_train = pnp.array(encoded_train, requires_grad=False)
    feats_test = encoded_test
    Y_train_pnp = pnp.array(Y_train.astype(np.float64), requires_grad=False)
    Y_test_np = np.asarray(Y_test)

    # Weights and optimizer
    weights_init = 0.01 * pnp.random.randn(num_layers, n_qubits, 3, requires_grad=True)
    bias_init = pnp.array(0.0, requires_grad=True)
    if optimizer == "adam":
        opt = AdamOptimizer(lr)
    else:
        opt = NesterovMomentumOptimizer(lr)

    # Compile (first run)
    t0 = time.perf_counter()
    _ = circuit(weights_init, feats_train[0])
    _ = cost(weights_init, bias_init, feats_train[:1], Y_train_pnp[:1])
    compile_sec = time.perf_counter() - t0

    # Optimize (batched steps; optional early stop every 100 steps)
    t0 = time.perf_counter()
    weights, bias = weights_init, bias_init
    steps_done = 0
    for step in range(iterations):
        batch_idx = rng.integers(0, n_train, size=(batch_size,))
        fb = feats_train[batch_idx]
        yb = Y_train_pnp[batch_idx]
        out = opt.step(cost, weights, bias, fb, yb)
        weights, bias = out[0], out[1]
        steps_done += 1
        if early_stop_target is not None and (step + 1) % 100 == 0:
            pred_test_now = np.sign(
                np.array([model(weights, bias, x) for x in pnp.array(feats_test)])
            ).flatten()
            test_acc_now = float(np.mean(np.abs(pred_test_now - Y_test_np) < 1e-5))
            if test_acc_now >= early_stop_target:
                break
    train_sec = time.perf_counter() - t0

    # Metrics
    pred_train = np.sign(
        np.array([model(weights, bias, x) for x in feats_train])
    ).flatten()
    pred_test = np.sign(
        np.array([model(weights, bias, x) for x in pnp.array(feats_test)])
    ).flatten()
    Y_train_np = np.array(Y_train_pnp)
    train_acc = float(np.mean(np.abs(pred_train - Y_train_np) < 1e-5))
    test_acc = float(np.mean(np.abs(pred_test - Y_test_np) < 1e-5))

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
        "qml_device": "cpu",
    }


def _run_training_gpu(
    encoded_train: torch.Tensor,
    encoded_test: torch.Tensor,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    *,
    n_qubits: int,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    early_stop_target: float | None = None,
    optimizer: str = "adam",
) -> dict[str, Any]:
    """GPU path: lightning.gpu + torch + adjoint + PyTorch optimizer.

    Encoded tensors stay on GPU (zero copy from QDP encoding).
    """
    wires = list(range(n_qubits))
    device = encoded_train.device
    real_dtype = (
        torch.float64 if encoded_train.dtype == torch.complex128 else torch.float32
    )
    Y_train_t = torch.tensor(Y_train, dtype=real_dtype, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=real_dtype, device=device)

    n_train = len(Y_train)
    rng = np.random.default_rng(seed)

    dev_qml = qml.device("lightning.gpu", wires=n_qubits)

    @qml.qnode(dev_qml, interface="torch", diff_method="adjoint")
    def circuit(weights, state_vector):
        qml.StatePrep(state_vector, wires=wires)
        for lw in weights:
            layer(lw, wires)
        return qml.expval(qml.PauliZ(0))

    def model(weights, bias, state_batch):
        return circuit(weights, state_batch) + bias

    def cost(weights, bias, X_batch, Y_batch):
        preds = model(weights, bias, X_batch)
        return torch.mean((Y_batch - preds) ** 2)

    torch.manual_seed(seed)
    weights = (
        (0.01 * torch.randn(num_layers, n_qubits, 3, device=device, dtype=real_dtype))
        .detach()
        .requires_grad_(True)
    )
    bias = (
        torch.zeros(1, device=device, dtype=real_dtype).squeeze().requires_grad_(True)
    )

    if optimizer == "adam":
        opt = torch.optim.Adam([weights, bias], lr=lr)
    else:
        opt = torch.optim.SGD([weights, bias], lr=lr, momentum=0.9, nesterov=True)

    # Compile (first run)
    t0 = time.perf_counter()
    _ = circuit(weights, encoded_train[0:1])
    _ = cost(weights, bias, encoded_train[:1], Y_train_t[:1])
    compile_sec = time.perf_counter() - t0

    # Optimize
    t0 = time.perf_counter()
    steps_done = 0
    for step in range(iterations):
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

    # Metrics
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
        "qml_device": "cuda",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QDP SVHN IQP variational classifier pipeline (2-class)"
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
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
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
    parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA device (default: 0)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=("cpu", "gpu"),
        help="Training backend: cpu (default.qubit) or gpu (lightning.gpu)",
    )
    parser.add_argument("--data-home", type=str, default=None, help="Data cache dir")
    args = parser.parse_args()

    n_qubits = args.n_qubits

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the QDP pipeline benchmark.")

    print("SVHN IQP variational classifier — QDP pipeline (2-class)")
    print(
        f"  {n_qubits} qubits, PCA 3072->{n_qubits}, "
        f"binary: digit {CLASS_POS} vs {CLASS_NEG}"
    )
    print(
        f"  n_samples={args.n_samples}, iters={args.iters}, "
        f"batch_size={args.batch_size}, layers={args.layers}, "
        f"lr={args.lr}, optimizer={args.optimizer}"
    )
    print(
        f"  CUDA: {torch.cuda.is_available()}, device_id: {args.device_id}, "
        f"backend={args.backend}"
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
    print(f"  Binary filtered: {len(Y_bin):,} samples (pos={np.mean(Y_bin == 1):.2f})")

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

    # Train/test split (done once before trials, like iris_amplitude QDP)
    n = len(Y_bin)
    split_rng = np.random.default_rng(args.seed)
    split_idx = split_rng.permutation(n)
    n_train = int(n * (1 - args.test_size))
    train_idx, test_idx = split_idx[:n_train], split_idx[n_train:]
    X_train_pca = X_pca[train_idx]
    X_test_pca = X_pca[test_idx]
    Y_train = Y_bin[train_idx]
    Y_test = Y_bin[test_idx]

    # QDP IQP encoding (one-time, GPU)
    X_train_params = features_to_iqp_params(X_train_pca, n_qubits)
    X_test_params = features_to_iqp_params(X_test_pca, n_qubits)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    encoded_train = encode_qdp(X_train_params, n_qubits, args.device_id)
    encoded_test = encode_qdp(X_test_params, n_qubits, args.device_id)
    torch.cuda.synchronize()

    if args.backend == "gpu":
        # Zero copy: QDP CUDA tensor -> lightning.gpu StatePrep (stays on GPU)
        encoded_train_data = encoded_train
        encoded_test_data = encoded_test
        encode_sec = time.perf_counter() - t0
    else:
        # CPU path: D2H copy required
        encoded_train_data = encoded_train.cpu().numpy()
        encoded_test_data = encoded_test.cpu().numpy()
        encode_sec = time.perf_counter() - t0  # includes D2H transfer

    state_dim = 2**n_qubits
    print(
        f"  IQP Encode (QDP): {encode_sec:.4f}s  "
        f"(n={n}, dim={state_dim}, "
        f"device={encoded_train.device}, dtype={encoded_train.dtype})"
    )

    if args.backend == "cpu":
        del encoded_train, encoded_test
        torch.cuda.empty_cache()

    print()

    results: list[dict[str, Any]] = []
    early_stop = args.early_stop if args.early_stop > 0 else None
    for t in range(args.trials):
        r = run_training(
            encoded_train_data,
            encoded_test_data,
            Y_train,
            Y_test,
            n_qubits=n_qubits,
            num_layers=args.layers,
            iterations=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + t,
            early_stop_target=early_stop,
            optimizer=args.optimizer,
            backend=args.backend,
        )
        results.append(r)
        print(f"  Trial {t + 1}:")
        print(f"    Compile:   {r['compile_time_sec']:.4f} s")
        print(f"    Train:     {r['train_time_sec']:.4f} s")
        print(f"    Train acc: {r['train_accuracy']:.4f}  (n={r['n_train']})")
        print(f"    Test acc:  {r['test_accuracy']:.4f}  (n={r['n_test']})")
        print(f"    Throughput: {r['samples_per_sec']:.1f} samples/s")
        print(f"    QML device: {r.get('qml_device', 'cpu')}")
        print()

    if args.backend == "gpu":
        # encoded_train_data / encoded_test_data are aliases for
        # encoded_train / encoded_test (zero copy); delete all refs.
        del encoded_train_data, encoded_test_data, encoded_train, encoded_test
        torch.cuda.empty_cache()

    if args.trials > 1:
        test_accs = [r["test_accuracy"] for r in results]
        print(
            f"  Best test accuracy:  {max(test_accs):.4f}  "
            f"(median: {float(np.median(test_accs)):.4f}, "
            f"mean: {float(np.mean(test_accs)):.4f} +/- {float(np.std(test_accs)):.4f}, "
            f"min: {min(test_accs):.4f}, max: {max(test_accs):.4f})"
        )

    # Summary: encoding vs training time
    avg_train = np.mean([r["train_time_sec"] for r in results])
    print(f"\n  {'─' * 50}")
    print(f"  PCA time:           {pca_sec:.4f}s")
    print(f"  IQP encode time:    {encode_sec:.4f}s  (one-time, QDP GPU)")
    print(f"  Avg train time:     {avg_train:.4f}s  (per trial)")
    print(
        f"  Encode fraction:    "
        f"{encode_sec / (encode_sec + avg_train) * 100:.1f}%  "
        f"(of encode + train)"
    )
    print(f"  Training backend:   {args.backend}")


if __name__ == "__main__":
    main()
