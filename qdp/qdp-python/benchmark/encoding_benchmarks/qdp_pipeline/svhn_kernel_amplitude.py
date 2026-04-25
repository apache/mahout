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
Quantum Kernel SVM — QDP pipeline (GPU encoding) — SVHN dataset.

Pipeline:
  SVHN (32×32×3) → Flatten (3072) → QdpEngine.encode(amplitude) on GPU (4096, 12 qubits)
    → Quantum Kernel K[i,j] = (encoded[i] · encoded[j])² → sklearn SVM

Encoding: QdpEngine (GPU) — data stays on GPU for kernel matmul, then moves to CPU for SVM.
Kernel:   Precomputed squared inner product (GPU torch.mm).
Classifier: sklearn.svm.SVC(kernel='precomputed').

Each pipeline step is timed separately to show the encoding fraction.
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request

import numpy as np
import torch

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: uv sync --group benchmark"
    ) from e

try:
    from scipy.io import loadmat
except ImportError as e:
    raise SystemExit("scipy is required. Install with: pip install scipy") from e

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
# Encoding & kernel
# ---------------------------------------------------------------------------

NUM_QUBITS = 12
STATE_DIM = 2**NUM_QUBITS  # 4096
CLASS_POS = 1
CLASS_NEG = 7


def _filter_binary(X, Y):
    mask = (Y == CLASS_POS) | (Y == CLASS_NEG)
    return X[mask], np.where(Y[mask] == CLASS_POS, 1, -1)


def encode_qdp(
    X: np.ndarray, device_id: int = 0, qdp_backend: str = "cuda"
) -> torch.Tensor:
    """QdpEngine amplitude encode → GPU float64 tensor (n, 4096)."""
    engine = QdpEngine(device_id=device_id, precision="float64", backend=qdp_backend)
    qt = engine.encode(
        X.astype(np.float64),
        num_qubits=NUM_QUBITS,
        encoding_method="amplitude",
    )
    encoded = torch.from_dlpack(qt)
    if encoded.is_complex():
        encoded = encoded.real
    return encoded[: X.shape[0]]


def compute_kernel_gpu(X1: torch.Tensor, X2: torch.Tensor) -> np.ndarray:
    """Quantum kernel on GPU: K[i,j] = (X1 @ X2.T)². Returns CPU numpy."""
    K = torch.mm(X1, X2.T)
    K = K**2
    return K.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantum Kernel SVM — QDP pipeline (GPU) — SVHN (12 qubits)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Total samples for CV (default: 5000)",
    )
    parser.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=100.0,
        help="SVM regularisation C (default: 100.0)",
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
    parser.add_argument("--data-home", type=str, default=None, help="Data cache dir")
    args = parser.parse_args()

    print("Quantum Kernel SVM — QDP pipeline (GPU) — SVHN")
    print(
        f"  {NUM_QUBITS} qubits, {STATE_DIM}-dim state, binary: digit {CLASS_POS} vs {CLASS_NEG}"
    )
    print(f"  n_samples={args.n_samples}, {args.folds}-fold CV, C={args.svm_c}")
    print(
        f"  GPU available: {torch.cuda.is_available()}, device_id: {args.device_id}, "
        f"qdp_backend: {args.qdp_backend}"
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
    print()

    # Step 1: StandardScaler + Encode (GPU)
    t0 = time.perf_counter()
    scaler = StandardScaler().fit(X_bin)
    X_scaled = scaler.transform(X_bin)
    X_encoded = encode_qdp(X_scaled, args.device_id, args.qdp_backend)
    torch.cuda.synchronize()
    encode_sec = time.perf_counter() - t0
    print(
        f"  Step 1: Scale+Encode ........  {encode_sec:.4f}s  (n={len(Y_bin)}, dim={STATE_DIM}, device={X_encoded.device})"
    )

    # Step 2: Full kernel matrix (GPU matmul → CPU)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    K_full = compute_kernel_gpu(X_encoded, X_encoded)
    kernel_sec = time.perf_counter() - t0
    print(
        f"  Step 2: Kernel      ........  {kernel_sec:.4f}s  ({K_full.shape[0]}×{K_full.shape[1]})"
    )

    # Free GPU memory
    del X_encoded
    torch.cuda.empty_cache()

    # Step 3: k-fold cross-validation
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_accs = []
    cv_fit_sec = 0.0
    cv_pred_sec = 0.0

    print(f"\n  Step 3: {args.folds}-fold Cross-Validation")
    for fold, (train_idx, test_idx) in enumerate(skf.split(K_full, Y_bin), 1):
        K_train = K_full[np.ix_(train_idx, train_idx)]
        K_test = K_full[np.ix_(test_idx, train_idx)]

        t0 = time.perf_counter()
        svm = SVC(kernel="precomputed", C=args.svm_c)
        svm.fit(K_train, Y_bin[train_idx])
        cv_fit_sec += time.perf_counter() - t0

        t0 = time.perf_counter()
        acc = svm.score(K_test, Y_bin[test_idx])
        cv_pred_sec += time.perf_counter() - t0

        fold_accs.append(acc)
        n_sv = svm.n_support_.sum()
        print(
            f"    Fold {fold}/{args.folds}: acc={acc:.4f}  "
            f"(train={len(train_idx)}, test={len(test_idx)}, SVs={n_sv})"
        )

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    total_sec = encode_sec + kernel_sec + cv_fit_sec + cv_pred_sec
    encode_pct = encode_sec / total_sec * 100

    print(f"\n  {'─' * 50}")
    print(f"  Encode time:        ........  {encode_sec:.4f}s")
    print(f"  Kernel time:        ........  {kernel_sec:.4f}s")
    print(f"  CV fit time:        ........  {cv_fit_sec:.4f}s  ({args.folds} folds)")
    print(f"  CV predict time:    ........  {cv_pred_sec:.4f}s")
    print(f"  Total:              ........  {total_sec:.4f}s")
    print(f"  Encoding fraction:  ........  {encode_pct:.1f}%")
    print(f"  Accuracy:           ........  {mean_acc:.4f} ± {std_acc:.4f}")


if __name__ == "__main__":
    main()
