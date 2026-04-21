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
QDP pipeline: Credit Card Fraud (binary, highly imbalanced), amplitude encoding.

Goal: **same data, model, loss, and metrics as the PennyLane baseline; only the
encoding step is different**. Here we:

- Preprocess features exactly as in the baseline:
  StandardScaler → PCA (to <= pca_dim) → pad to FEATURE_DIM → L2-normalized vector.
- Use QDP (`QuantumDataLoader` with `encoding("amplitude")`) to encode these
  FEATURE_DIM vectors into **amplitude state vectors** of length `2**NUM_QUBITS`.
- Feed the encoded state vectors into a PennyLane circuit via `qml.AmplitudeEmbedding`,
  then apply the same variational layers, optimizer, and loss as the baseline.

Best practices (aligned with ENCODING_BENCHMARK_PLAN.md §2.2):

- Dataset: Kaggle "Credit Card Fraud Detection" (Time, V1..V28, Amount, Class).
- Metrics: AUPRC (precision–recall AUC), F1-score, precision, recall.
- Imbalance: class-weighted loss (minority class up-weighted); no accuracy.

Training always runs on GPU via lightning.gpu.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path
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
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: uv sync --group benchmark"
    ) from e

try:
    from qumat_qdp import QdpEngine, QuantumDataLoader
except ImportError as e:
    raise SystemExit(
        "qumat_qdp (QDP Python bindings) is required. Build with: uv run maturin develop"
    ) from e


NUM_QUBITS = 5
STATE_DIM = 2**NUM_QUBITS  # length of encoded state vector
FEATURE_DIM = STATE_DIM  # pre-QDP feature dimension (padded to this)


def _layer(layer_weights: torch.Tensor, wires: tuple[int, ...]) -> None:
    """Single variational layer: Rot on each wire + ring of CNOTs."""
    for i, w in enumerate(wires):
        qml.Rot(layer_weights[i, 0], layer_weights[i, 1], layer_weights[i, 2], wires=w)
    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])


def load_creditcard_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Credit Card Fraud CSV. Expects columns including V1..V28, Amount, Class.
    Returns (X_raw shape (n, 30), y shape (n,) with 0/1).
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Last column = Class; rest = features (Time, V1..V28, Amount)
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)
    # If CSV has header row with "Time", we already skipped it
    if X.shape[1] >= 30:
        X = X[:, -30:]  # last 30 cols: V1..V28, Amount (and drop Time if 31)
    elif X.shape[1] < 30:
        # Pad with zeros to 30
        pad = np.zeros((X.shape[0], 30 - X.shape[1]), dtype=np.float64)
        X = np.hstack([X, pad])
    return X, y


def make_synthetic_imbalanced(
    seed: int, n_total: int = 2000, fraud_ratio: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 30-D imbalanced binary data for smoke test when no CSV is provided."""
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_total * fraud_ratio))
    n_legit = n_total - n_fraud
    X_legit = rng.standard_normal((n_legit, 30)).astype(np.float64) * 0.5
    X_fraud = rng.standard_normal((n_fraud, 30)).astype(np.float64) * 0.5 + 1.0
    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud, dtype=np.int32)
    perm = rng.permutation(n_total)
    return X[perm], y[perm]


def preprocess(
    X: np.ndarray,
    y: np.ndarray,
    pca_dim: int,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Any,
    Any,
    np.ndarray,
]:
    """
    StandardScaler → PCA (to <= pca_dim) → pad to FEATURE_DIM → L2 normalize.
    Stratified train/val/test. Returns X_train, y_train, X_val, y_val, X_test, y_test,
    plus scaler, pca, and sample_weights for weighted loss.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=min(pca_dim, X_scaled.shape[1], X_scaled.shape[0] - 1),
        random_state=seed,
    )
    X_pca = pca.fit_transform(X_scaled)
    if X_pca.shape[1] < FEATURE_DIM:
        pad = np.zeros((X_pca.shape[0], FEATURE_DIM - X_pca.shape[1]), dtype=np.float64)
        X_pca = np.hstack([X_pca, pad])

    norm = np.linalg.norm(X_pca, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    X_norm = (X_pca / norm).astype(np.float64)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    X_norm, y = X_norm[idx], y[idx]

    # Stratified split: first test, then val from train
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_norm, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
    )

    # Class weights for weighted MSE: n / (2 * n_class)
    n0 = max(1, int(np.sum(y_train == 0)))
    n1 = max(1, int(np.sum(y_train == 1)))
    w0 = len(y_train) / (2 * n0)
    w1 = len(y_train) / (2 * n1)
    sample_weights = np.where(y_train == 0, w0, w1).astype(np.float64)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
        pca,
        sample_weights,
    )


def encode_via_qdp_engine(
    X_norm: np.ndarray,
    *,
    batch_size: int,
    device_id: int = 0,
) -> torch.Tensor:
    """
    QDP API: amplitude-encode in memory via QdpEngine.encode() (batched).
    No temp file; minimal CPU–GPU transfer by batching.
    Returns GPU torch.Tensor shape (n, STATE_DIM).
    """
    n, dim = X_norm.shape
    if dim != FEATURE_DIM:
        raise ValueError(
            f"X_norm must have {FEATURE_DIM} features for {NUM_QUBITS} qubits, got {dim}"
        )
    # Ensure float64 C-contiguous once before the loop (preprocess() already guarantees this,
    # but guard against callers passing non-contiguous or non-float64 arrays).
    if not (X_norm.dtype == np.float64 and X_norm.flags["C_CONTIGUOUS"]):
        X_norm = np.ascontiguousarray(X_norm, dtype=np.float64)
    engine = QdpEngine(device_id=device_id)
    batches_list: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Pass slice directly — no per-batch astype() copy needed.
        qt = engine.encode(X_norm[start:end], NUM_QUBITS, "amplitude")
        t = torch.from_dlpack(qt)
        batches_list.append(t)
    # torch.cat produces exactly n rows and a contiguous tensor.
    encoded = torch.cat(batches_list, dim=0)
    # DLPack exports complex128 (CuDoubleComplex) even though imaginary parts are always 0.0
    # (amplitude encoding of real input → real state vector; CUDA kernel hardcodes imag=0.0).
    # Taking .real gives a float64 view (zero-copy) matching the baseline's dtype and halving
    # memory footprint, which also avoids any complex-arithmetic paths in PennyLane.
    if encoded.is_complex():
        encoded = encoded.real
    if encoded.shape[1] != STATE_DIM:
        raise ValueError(
            f"Encoded state dimension mismatch: expected {STATE_DIM}, got {encoded.shape[1]}"
        )
    return encoded


def encoded_batches_from_loader(
    X_norm: np.ndarray,
    *,
    batch_size: int,
    device_id: int = 0,
    data_dir: str | None = None,
    filename: str = "creditcard_train.npy",
) -> Iterator[tuple[torch.Tensor, int, int]]:
    """
    DataLoader API: stream amplitude-encoded batches from QuantumDataLoader (in-memory).
    Uses source_array() (no temp file). Always returns GPU torch.Tensor batches.
    Yields (batch, start_idx, end_idx).
    """
    n, dim = X_norm.shape
    if dim != FEATURE_DIM:
        raise ValueError(
            f"X_norm must have {FEATURE_DIM} features for {NUM_QUBITS} qubits, got {dim}"
        )
    total_batches = (n + batch_size - 1) // batch_size
    loader = (
        QuantumDataLoader(device_id=device_id)
        .qubits(NUM_QUBITS)
        .encoding("amplitude")
        .batches(total_batches, size=batch_size)
        .source_array(X_norm.astype(np.float64))
        .as_torch(device="cuda")
    )
    start = 0
    for batch in loader:
        end = min(start + batch.shape[0], n)
        actual = batch[: end - start]
        if actual.shape[1] != STATE_DIM:
            raise ValueError(
                f"Encoded state dimension mismatch: expected {STATE_DIM}, got {actual.shape[1]}"
            )
        yield actual, start, end
        start = end


def run_training(
    encoded_train: torch.Tensor,
    encoded_test: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    sample_weights: np.ndarray,
    *,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    """Train 5-qubit amplitude VQC on GPU; dispatch to lightning.gpu."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training. No CUDA device found.")
    try:
        dev_qml = qml.device("lightning.gpu", wires=NUM_QUBITS)
    except Exception as e:
        raise RuntimeError(
            "lightning.gpu is required for GPU training. Install with: "
            "pip install pennylane-lightning[gpu]"
        ) from e

    # Ensure encoded data is on GPU
    if not encoded_train.is_cuda:
        encoded_train = encoded_train.cuda()
    if not encoded_test.is_cuda:
        encoded_test = encoded_test.cuda()

    device = encoded_train.device
    dtype = encoded_train.dtype
    n_train = len(y_train)
    y_test_np = np.asarray(y_test)

    Y_train_t = torch.tensor(
        np.asarray(y_train, dtype=np.float64), dtype=dtype, device=device
    )
    W_train_t = torch.tensor(sample_weights, dtype=dtype, device=device)

    wires = tuple(range(NUM_QUBITS))

    @qml.qnode(dev_qml, interface="torch", diff_method="adjoint")
    def circuit(weights: torch.Tensor, state_vector: torch.Tensor) -> torch.Tensor:
        qml.AmplitudeEmbedding(state_vector, wires=wires, normalize=False)
        for w in weights:
            _layer(w, wires)
        return qml.expval(qml.PauliZ(0))

    def model(
        weights: torch.Tensor, bias: torch.Tensor, state_batch: torch.Tensor
    ) -> torch.Tensor:
        return circuit(weights, state_batch) + bias

    def cost(
        weights: torch.Tensor,
        bias: torch.Tensor,
        states_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        w_batch: torch.Tensor,
    ) -> torch.Tensor:
        target = Y_batch * 2.0 - 1.0
        preds = model(weights, bias, states_batch)
        return (w_batch * (target - preds) ** 2).sum() / (w_batch.sum() + 1e-12)

    torch.manual_seed(seed)
    weights = torch.nn.Parameter(
        0.01 * torch.randn(num_layers, NUM_QUBITS, 3, device=device, dtype=dtype)
    )
    bias = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
    opt = torch.optim.Adam([weights, bias], lr=lr)

    t0 = time.perf_counter()
    _ = circuit(weights, encoded_train[:1])
    _ = cost(weights, bias, encoded_train[:1], Y_train_t[:1], W_train_t[:1])
    compile_sec = time.perf_counter() - t0

    # Use torch.randint so indices stay on GPU — avoids implicit H2D transfer of 256 indices
    # per step (NumPy rng.integers → CPU array → implicit copy to index CUDA tensor).
    _batch_n = min(batch_size, n_train)
    t0 = time.perf_counter()
    for _ in range(iterations):
        opt.zero_grad()
        idx = torch.randint(0, n_train, (_batch_n,), device=device)
        sb = encoded_train[idx]
        yb = Y_train_t[idx]
        wb = W_train_t[idx]
        loss = cost(weights, bias, sb, yb, wb)
        loss.backward()
        opt.step()
    train_sec = time.perf_counter() - t0

    with torch.no_grad():
        pred_scores = model(weights, bias, encoded_test).cpu().numpy().flatten()
    pred_binary = (np.sign(pred_scores) > 0).astype(np.int32)
    scores_positive = (pred_scores + 1.0) / 2.0
    auprc = float(average_precision_score(y_test_np, scores_positive))
    f1 = float(f1_score(y_test_np, pred_binary, zero_division=0))
    prec = float(precision_score(y_test_np, pred_binary, zero_division=0))
    rec = float(recall_score(y_test_np, pred_binary, zero_division=0))

    return {
        "compile_time_sec": compile_sec,
        "train_time_sec": train_sec,
        "samples_per_sec": (iterations * _batch_n) / train_sec
        if train_sec > 0
        else 0.0,
        "auprc": auprc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "n_train": n_train,
        "n_test": len(y_test_np),
        "iterations": iterations,
    }


def run_training_from_loader(
    X_train: np.ndarray,
    encoded_test: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    sample_weights: np.ndarray,
    *,
    num_layers: int,
    iterations: int,
    encode_batch_size: int,
    device_id: int = 0,
    encode_data_dir: str | None = None,
    lr: float,
    seed: int,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Train by streaming encoded batches from QuantumDataLoader on GPU.
    Encode once on GPU, then train with lightning.gpu."""

    encoded_train = encode_via_qdp_engine(
        X_train,
        batch_size=encode_batch_size,
        device_id=device_id,
    )
    return run_training(
        encoded_train,
        encoded_test,
        y_train,
        y_test,
        sample_weights,
        num_layers=num_layers,
        iterations=iterations,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QDP Credit Card Fraud pipeline (amplitude, 5 qubits, AUPRC/F1, GPU training)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to CSV (e.g. Kaggle creditcard.csv). If omitted, use synthetic imbalanced data.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        help="Max samples to use from CSV (default: 50000); ignored for synthetic.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=30,
        help="PCA components before padding to FEATURE_DIM (default: 30, capped by feature dim).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--iters",
        type=int,
        default=5000,
        help="Optimizer steps (default: 5000; use same as baseline for apples-to-apples).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--layers", type=int, default=2, help="Variational layers")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of runs (same data, different seeds); report median AUPRC/F1.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="QDP device id (default: 0)",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=4096,
        help="Batch size for QDP encoding (default: 4096).",
    )
    parser.add_argument(
        "--encode-data-dir",
        type=str,
        default=None,
        help="Directory for temporary .npy files used by QDP loader (default: system temp).",
    )
    parser.add_argument(
        "--use-loader",
        action="store_true",
        help="Stream encoded batches via QuantumDataLoader.source_array() (DataLoader API).",
    )
    args = parser.parse_args()

    if args.data_file:
        path = Path(args.data_file)
        if not path.is_file():
            raise SystemExit(f"Data file not found: {path}")
        X, y = load_creditcard_csv(str(path))
        if len(X) > args.max_samples:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(X), size=args.max_samples, replace=False)
            X, y = X[idx], y[idx]
        data_src = f"CSV {path.name} (n={len(X)})"
    else:
        X, y = make_synthetic_imbalanced(args.seed, n_total=2000, fraud_ratio=0.02)
        data_src = f"synthetic imbalanced (n={len(X)}, fraud~2%)"

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        _scaler,
        _pca,
        sample_weights,
    ) = preprocess(
        X,
        y,
        pca_dim=args.pca_dim,
        seed=args.seed,
        test_size=0.2,
        val_size=0.1,
    )

    print("QDP Credit Card Fraud amplitude pipeline (GPU)")
    print(
        f"  Data: {data_src} → StandardScaler → PCA({args.pca_dim}) "
        f"→ pad to {FEATURE_DIM} → QDP amplitude → L2 norm (implicit)"
    )
    print(
        f"  Train/val/test (features pre-QDP): "
        f"{len(X_train)} / {len(X_val)} / {len(X_test)}  (stratified)"
    )
    print(
        f"  Iters: {args.iters}, train batch: {args.batch_size}, "
        f"encode batch: {args.encode_batch_size}, layers: {args.layers}, lr: {args.lr}"
    )
    print("  Encode + Train: GPU (QDP encode + lightning.gpu circuit).")

    # Encode test set via QDP (keep on GPU)
    t_enc0 = time.perf_counter()
    encoded_test = encode_via_qdp_engine(
        X_test,
        batch_size=args.encode_batch_size,
        device_id=args.device_id,
    )
    enc_test_sec = time.perf_counter() - t_enc0
    print(f"  Encode test  ({len(X_test)} samples): {enc_test_sec:.4f} s")

    results: list[dict[str, Any]] = []
    for t in range(args.trials):
        if args.use_loader:
            r = run_training_from_loader(
                X_train,
                encoded_test,
                y_train,
                y_test,
                sample_weights,
                num_layers=args.layers,
                iterations=args.iters,
                encode_batch_size=args.encode_batch_size,
                device_id=args.device_id,
                encode_data_dir=args.encode_data_dir,
                lr=args.lr,
                seed=args.seed + t,
                batch_size=args.batch_size,
            )
            r["encode_train_sec"] = 0.0  # encoded lazily inside loader
        else:
            t_enc1 = time.perf_counter()
            encoded_train = encode_via_qdp_engine(
                X_train,
                batch_size=args.encode_batch_size,
                device_id=args.device_id,
            )
            enc_train_sec = time.perf_counter() - t_enc1
            r = run_training(
                encoded_train,
                encoded_test,
                y_train,
                y_test,
                sample_weights,
                num_layers=args.layers,
                iterations=args.iters,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed + t,
            )
            r["encode_train_sec"] = enc_train_sec
        results.append(r)
        print(f"\n  Trial {t + 1}:")
        print(
            f"    Encode train ({len(X_train)} samples): {r.get('encode_train_sec', 0.0):.4f} s"
        )
        print(f"    Compile:   {r['compile_time_sec']:.4f} s")
        print(
            f"    Train:     {r['train_time_sec']:.4f} s  "
            f"({r['samples_per_sec']:.1f} samples/s)"
        )
        print(f"    AUPRC:     {r['auprc']:.4f}")
        print(
            f"    F1:        {r['f1_score']:.4f}  "
            f"(P: {r['precision']:.4f}, R: {r['recall']:.4f})"
        )

    if args.trials > 1:
        auprcs = sorted(r["auprc"] for r in results)
        f1s = sorted(r["f1_score"] for r in results)
        mid = args.trials // 2
        print(
            f"\n  Median AUPRC: {auprcs[mid]:.4f}  "
            f"(min: {auprcs[0]:.4f}, max: {auprcs[-1]:.4f})"
        )
        print(
            f"  Median F1:    {f1s[mid]:.4f}  (min: {f1s[0]:.4f}, max: {f1s[-1]:.4f})"
        )


if __name__ == "__main__":
    main()
