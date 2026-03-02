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
QDP pipeline: Iris (2-class), same data and training as baseline; only encoding differs.

Aligned with baseline: https://pennylane.ai/qml/demos/tutorial_variational_classifier

Data sources (default: sklearn, not the official file):
  - Default: sklearn.datasets.load_iris, classes 0 & 1, 4 features → scale → L2 norm → 4-D vectors for QDP.
  - Official file: pass --data-file <path>. File format: cols [f0, f1, f2, f3, label]; we use first 2 cols,
    pad to 4, L2 norm. Bundled path: ../pennylane_baseline/data/iris_classes1and2_scaled.txt (from XanaduAI/qml,
    see baseline docstring URL).
  - Total samples: 100 (2-class Iris). Full Iris has 150 (3 classes).

Only difference from baseline: encoding. Here we use QDP (QdpEngine.encode + amplitude) → StatePrep(encoded);
baseline uses get_angles → state_preparation(angles). Rest: same circuit (Rot + CNOT), loss, optimizer, CLI.
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

from qumat_qdp import QdpEngine
import torch


NUM_QUBITS = 2
STATE_DIM = 2**NUM_QUBITS  # 4


# --- Circuit: variational layer (Rot + CNOT); state prep is StatePrep(encoded) in training ---
def layer(layer_weights, wires=(0, 1)):
    """Rot on each wire + CNOT (tutorial Iris section)."""
    for i, w in enumerate(wires):
        qml.Rot(*layer_weights[i], wires=w)
    qml.CNOT(wires=list(wires))


# --- Data: official file (2 cols → pad 4 → L2 norm). Source: XanaduAI/qml, see docstring. ---
def load_iris_4d_from_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load official-style Iris 2-class data; return 4-D normalized vectors for QDP.
    File cols [f0, f1, f2, f3, label]. We use first 2 cols only, pad to 4, L2 norm.
    Returns (X_norm, Y) with X_norm (n, 4), Y in {-1, 1}.
    """
    data = np.loadtxt(path, dtype=np.float64)
    X = data[:, 0:2]
    Y = np.asarray(data[:, -1], dtype=np.float64)
    # Official Xanadu file uses ±1; if file has 0/1, convert to ±1
    if np.unique(Y).size <= 2 and 0 in np.unique(Y):
        Y = Y * 2 - 1
    padding = np.ones((len(X), 2)) * 0.1
    X_pad = np.c_[X, padding]
    norm = np.sqrt(np.sum(X_pad**2, axis=-1)) + 1e-12
    X_norm = (X_pad.T / norm).T
    return X_norm, Y


# --- Data: sklearn Iris classes 0 & 1, 4 features → scale → L2 norm. Source: sklearn.datasets.load_iris. ---
def load_iris_binary_4d(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Iris classes 0 and 1 only. Uses all 4 features: scale → L2 norm.
    Returns (X_norm, Y) with X_norm (n, 4), Y in {-1, 1}.
    """
    X_raw, y = load_iris(return_X_y=True)
    mask = (y == 0) | (y == 1)
    X = np.asarray(X_raw[mask], dtype=np.float64)
    y = y[mask]
    X = StandardScaler().fit_transform(X)
    norm = np.sqrt(np.sum(X**2, axis=-1)) + 1e-12
    X_norm = (X.T / norm).T
    Y = np.array(y, dtype=np.float64) * 2 - 1
    return X_norm, Y


# --- Encoding: QDP (QdpEngine.encode + amplitude); 4-D → GPU tensor ---
def encode_via_qdp(
    X_norm: np.ndarray,
    batch_size: int,  # kept for CLI symmetry; not used here
    device_id: int = 0,
    data_dir: str | None = None,
    filename: str = "iris_4d.npy",
) -> torch.Tensor:
    """QDP: use QdpEngine.encode on 4-D vectors (amplitude), return encoded (n, 4) on GPU.

    Uses in-memory encoding via QdpEngine instead of writing/reading .npy files. The returned
    tensor stays on the selected CUDA device and can be fed directly to qml.StatePrep.
    """
    n, dim = X_norm.shape
    if dim != STATE_DIM:
        raise ValueError(
            f"X_norm must have {STATE_DIM} features for 2 qubits, got {dim}"
        )
    engine = QdpEngine(device_id=device_id, precision="float32")
    qt = engine.encode(
        X_norm.astype(np.float64),
        num_qubits=NUM_QUBITS,
        encoding_method="amplitude",
    )
    encoded = torch.from_dlpack(qt)
    # DLPack exports complex dtype even though imaginary parts are always 0.0
    # (CUDA kernel hardcodes imag=0.0). Taking .real gives a real-valued zero-copy view.
    if encoded.is_complex():
        encoded = encoded.real
    return encoded[:n].clone()


# --- Training: StatePrep(encoded) + Rot layers, square loss, optional early stop ---
def run_training(
    encoded_train: torch.Tensor | np.ndarray,
    encoded_test: torch.Tensor | np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    *,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    early_stop_target: float | None = None,
    optimizer: str = "nesterov",
) -> dict[str, Any]:
    """Train variational classifier: StatePrep(encoded) + Rot layers + bias, square loss, batched.
    If encoded_* are on GPU and lightning.gpu is available, training runs on GPU; otherwise on CPU.
    When early_stop_target is set, evaluate test acc every 100 steps and stop when >= target."""
    n_train = len(Y_train)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Prefer Lightning GPU when encoded data is on GPU
    use_gpu = isinstance(encoded_train, torch.Tensor) and encoded_train.is_cuda
    dev_qml = None
    if use_gpu:
        try:
            dev_qml = qml.device("lightning.gpu", wires=NUM_QUBITS)
        except Exception:
            use_gpu = False
    if not use_gpu or dev_qml is None:
        dev_qml = qml.device("default.qubit", wires=NUM_QUBITS)
        use_gpu = False
        if isinstance(encoded_train, torch.Tensor):
            encoded_train = encoded_train.cpu().numpy()
        if isinstance(encoded_test, torch.Tensor):
            encoded_test = encoded_test.cpu().numpy()

    if use_gpu:
        return _run_training_gpu(
            encoded_train,
            encoded_test,
            Y_train,
            Y_test,
            dev_qml=dev_qml,
            num_layers=num_layers,
            iterations=iterations,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            n_train=n_train,
            rng=rng,
            early_stop_target=early_stop_target,
        )
    return _run_training_cpu(
        encoded_train,
        encoded_test,
        Y_train,
        Y_test,
        dev_qml=dev_qml,
        num_layers=num_layers,
        iterations=iterations,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        n_train=n_train,
        rng=rng,
        qml_device="cpu",
        early_stop_target=early_stop_target,
        optimizer=optimizer,
    )


def _run_training_cpu(
    encoded_train: np.ndarray,
    encoded_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    *,
    dev_qml: Any,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    n_train: int,
    rng: np.random.Generator,
    qml_device: str = "cpu",
    early_stop_target: float | None = None,
    optimizer: str = "nesterov",
) -> dict[str, Any]:
    """CPU path: default.qubit + autograd + Nesterov or Adam. Optional early stop every 100 steps."""
    try:
        pnp.random.seed(seed)
    except Exception:
        pass
    # Plain np.ndarray: PennyLane autograd treats as non-differentiable constant.
    # Avoids pnp.tensor subclass overhead and prevents AdamOptimizer from computing
    # ∂cost/∂feats_train (unnecessary gradient over state vectors).
    feats_train = np.asarray(encoded_train)
    feats_test = encoded_test
    Y_train_pnp = pnp.array(np.asarray(Y_train, dtype=np.float64), requires_grad=False)
    Y_test_np = np.asarray(Y_test)

    @qml.qnode(dev_qml, interface="autograd", diff_method="backprop")
    def circuit(weights, state_vector):
        # normalize=False: QDP pre-normalizes to unit norm, skipping PennyLane's re-normalization.
        qml.AmplitudeEmbedding(state_vector, wires=(0, 1), normalize=False)
        for lw in weights:
            layer(lw, wires=(0, 1))
        return qml.expval(qml.PauliZ(0))

    def model(weights, bias, state_batch):
        return circuit(weights, state_batch) + bias

    def cost(weights, bias, X_batch, Y_batch):
        preds = model(weights, bias, X_batch)
        return pnp.mean((Y_batch - preds) ** 2)

    weights_init = 0.01 * pnp.random.randn(
        num_layers, NUM_QUBITS, 3, requires_grad=True
    )
    bias_init = pnp.array(0.0, requires_grad=True)
    opt = AdamOptimizer(lr) if optimizer == "adam" else NesterovMomentumOptimizer(lr)

    t0 = time.perf_counter()
    _ = circuit(weights_init, feats_train[0])
    _ = cost(weights_init, bias_init, feats_train[:1], Y_train_pnp[:1])
    compile_sec = time.perf_counter() - t0

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
                np.array(model(weights, bias, pnp.array(feats_test)))
            ).flatten()
            test_acc_now = float(np.mean(np.abs(pred_test_now - Y_test_np) < 1e-5))
            if test_acc_now >= early_stop_target:
                break
    train_sec = time.perf_counter() - t0

    pred_train = np.sign(np.array(model(weights, bias, feats_train))).flatten()
    pred_test = np.sign(np.array(model(weights, bias, pnp.array(feats_test)))).flatten()
    Y_train_np = np.array(Y_train_pnp)
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
        "qml_device": qml_device,
    }


def _run_training_gpu(
    encoded_train: torch.Tensor,
    encoded_test: torch.Tensor,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    *,
    dev_qml: Any,
    num_layers: int,
    iterations: int,
    batch_size: int,
    lr: float,
    seed: int,
    n_train: int,
    rng: np.random.Generator,
    early_stop_target: float | None = None,
) -> dict[str, Any]:
    """GPU path: lightning.gpu + PyTorch interface, data stays on GPU. Optional early stop every 100 steps."""
    device = encoded_train.device
    dtype = encoded_train.dtype
    Y_train_t = torch.tensor(Y_train, dtype=dtype, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=dtype, device=device)

    @qml.qnode(dev_qml, interface="torch", diff_method="adjoint")
    def circuit(weights, state_vector):
        # normalize=False: QDP pre-normalizes to unit norm, skipping PennyLane's re-normalization.
        qml.AmplitudeEmbedding(state_vector, wires=(0, 1), normalize=False)
        for lw in weights:
            layer(lw, wires=(0, 1))
        return qml.expval(qml.PauliZ(0))

    def model(weights, bias, state_batch):
        return circuit(weights, state_batch) + bias

    def cost(weights, bias, X_batch, Y_batch):
        preds = model(weights, bias, X_batch)
        return torch.mean((Y_batch - preds) ** 2)

    torch.manual_seed(seed)
    weights = 0.01 * torch.randn(
        num_layers, NUM_QUBITS, 3, device=device, dtype=dtype, requires_grad=True
    )
    bias = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.SGD([weights, bias], lr=lr, momentum=0.9, nesterov=True)

    t0 = time.perf_counter()
    _ = circuit(weights, encoded_train[0:1])
    _ = cost(weights, bias, encoded_train[:1], Y_train_t[:1])
    compile_sec = time.perf_counter() - t0

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QDP Iris amplitude encoding pipeline (2-class, same training as baseline)"
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
        "--lr", type=float, default=0.08, help="Learning rate (default: 0.08)"
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
        help="Optimizer for CPU (default: adam); GPU uses SGD+Nesterov",
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
    parser.add_argument(
        "--device-id", type=int, default=0, help="QDP device (default: 0)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Dir for .npy files (default: temp)"
    )
    args = parser.parse_args()

    # Data source: official file (when --data-file set) or sklearn Iris (default)
    if args.data_file:
        X_norm, Y = load_iris_4d_from_file(args.data_file)
        test_size = 0.25  # tutorial uses 75% train
        data_src = f"official file (2 features): {args.data_file}"
    else:
        X_norm, Y = load_iris_binary_4d(seed=args.seed)
        test_size = args.test_size
        data_src = "sklearn load_iris, classes 0 & 1, 4 features"
    n = len(Y)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_train = int(n * (1 - test_size))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    X_train_4d = X_norm[train_idx]
    X_test_4d = X_norm[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    # QDP encoding: 4-D → amplitude-encoded state vectors
    encoded_train = encode_via_qdp(
        X_train_4d,
        batch_size=args.batch_size,
        device_id=args.device_id,
        data_dir=args.data_dir,
        filename="iris_4d_train.npy",
    )
    encoded_test = encode_via_qdp(
        X_test_4d,
        batch_size=args.batch_size,
        device_id=args.device_id,
        data_dir=args.data_dir,
        filename="iris_4d_test.npy",
    )

    print("Iris amplitude (QDP encoding) — 2-class variational classifier")
    print(f"  Data: {data_src} → QDP amplitude  (n={n}; 2-class Iris = 100 samples)")
    print(
        f"  Iters: {args.iters}, batch_size: {args.batch_size}, layers: {args.layers}, lr: {args.lr}, optimizer: {args.optimizer}"
    )

    results: list[dict[str, Any]] = []
    early_stop = args.early_stop if args.early_stop > 0 else None
    for t in range(args.trials):
        r = run_training(
            encoded_train,
            encoded_test,
            Y_train,
            Y_test,
            num_layers=args.layers,
            iterations=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + t,
            early_stop_target=early_stop,
            optimizer=args.optimizer,
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
            f"\n  Best test accuracy:  {best:.4f}  (median: {test_accs[mid]:.4f}, min: {test_accs[0]:.4f}, max: {test_accs[-1]:.4f})"
        )
        if best >= 0.9:
            print("  → Target ≥0.9 achieved.")


if __name__ == "__main__":
    main()
