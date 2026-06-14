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
FINAL END-TO-END BENCHMARK (Disk -> GPU VRAM).

Scope:
1. Disk IO: Reading Parquet file.
2. Preprocessing: L2 Normalization (CPU vs GPU).
3. Encoding: Quantum State Preparation.
4. Transfer: Moving data to GPU VRAM.
5. Consumption: 1 dummy Forward Pass to ensure data is usable.

This is the most realistic comparison for a "Cold Start" Training Epoch.
"""

import argparse
import gc
import itertools
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from _qdp import QdpEngine
from utils import generate_batch_data, normalize_batch

# Competitors
try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

# Config
DATA_FILE = "final_benchmark_data.parquet"
ARROW_FILE = "final_benchmark_data.arrow"
HIDDEN_DIM = 16
BATCH_SIZE = 64  # Small batch to stress loop overhead


def _input_dim(n_qubits: int, encoding_method: str) -> int:
    """Return per-sample input dimension for the encoding method."""
    if encoding_method == "angle":
        return n_qubits
    if encoding_method == "iqp-z":
        return n_qubits
    if encoding_method == "iqp":
        return n_qubits + n_qubits * (n_qubits - 1) // 2
    return 1 << n_qubits


def _mahout_encode_batch(
    engine,
    data,
    n_qubits: int,
    encoding_method: str,
    encode_path: str,
):
    """Dispatch Mahout encode via FWT or Tensor Core path (GPU vs GPU)."""
    if encode_path == "tc":
        if encoding_method not in ("iqp", "iqp-z"):
            raise ValueError("encode_path=tc requires encoding_method iqp or iqp-z")
        if not hasattr(engine, "encode_batch_tc"):
            raise RuntimeError("encode_batch_tc not available in this build")
        return engine.encode_batch_tc(data, n_qubits)
    return engine.encode(data, n_qubits, encoding_method)


def clean_cache() -> None:
    """Clear GPU cache and Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DummyQNN(nn.Module):
    def __init__(self, n_qubits) -> None:
        super().__init__()
        self.fc = nn.Linear(1 << n_qubits, HIDDEN_DIM)

    def forward(self, x):
        return self.fc(x)


def generate_data(n_qubits, n_samples, encoding_method: str = "amplitude") -> None:
    for f in [DATA_FILE, ARROW_FILE]:
        if os.path.exists(f):
            os.remove(f)

    print(f"Generating {n_samples} samples of {n_qubits} qubits...")
    dim = _input_dim(n_qubits, encoding_method)

    # Generate all data at once
    all_data = generate_batch_data(n_samples, dim, encoding_method, seed=42)

    # Save as Parquet
    if encoding_method == "basis":
        # For basis encoding, save single scalar indices (not lists)
        table = pa.table({"index": pa.array(all_data.flatten(), type=pa.float64())})
    else:
        # For amplitude/angle encoding, use List format for PennyLane/Qiskit compatibility
        feature_vectors = [row.tolist() for row in all_data]
        table = pa.table(
            {"feature_vector": pa.array(feature_vectors, type=pa.list_(pa.float64()))}
        )
    pq.write_table(table, DATA_FILE)

    # Save as Arrow IPC (FixedSizeList format for Mahout)
    if encoding_method == "basis":
        # For basis encoding, use FixedSizeList(len=1) for Mahout Arrow reader compatibility
        arr = pa.FixedSizeListArray.from_arrays(pa.array(all_data.flatten()), 1)
        arrow_table = pa.table({"data": arr})
    else:
        # For amplitude/angle encoding, use FixedSizeList format
        arr = pa.FixedSizeListArray.from_arrays(pa.array(all_data.flatten()), dim)
        arrow_table = pa.table({"data": arr})
    with ipc.RecordBatchFileWriter(ARROW_FILE, arrow_table.schema) as writer:
        writer.write_table(arrow_table)

    parquet_size = os.path.getsize(DATA_FILE) / (1024 * 1024)
    arrow_size = os.path.getsize(ARROW_FILE) / (1024 * 1024)
    print(f"  Generated {n_samples} samples")
    print(f"  Parquet: {parquet_size:.2f} MB, Arrow IPC: {arrow_size:.2f} MB")

    # Clean cache after data generation
    clean_cache()


# -----------------------------------------------------------
# 1. Qiskit Full Pipeline
# -----------------------------------------------------------
def run_qiskit(n_qubits, n_samples, encoding_method: str = "amplitude"):
    if not HAS_QISKIT:
        print("\n[Qiskit] Not installed, skipping.")
        return 0.0, None

    # Clean cache before starting benchmark
    clean_cache()

    print(f"\n[Qiskit] Full Pipeline (Disk -> GPU) - {encoding_method} encoding...")
    model = DummyQNN(n_qubits).cuda()
    backend = AerSimulator(method="statevector")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # IO
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    if encoding_method == "basis":
        raw_data = df["index"].values.astype(np.int64)
    else:
        raw_data = np.stack(df["feature_vector"].values)
    io_time = time.perf_counter() - start_time
    print(f"  IO Time: {io_time:.4f} s")

    all_qiskit_states = []

    # Process batches
    for i in range(0, n_samples, BATCH_SIZE):
        batch = normalize_batch(raw_data[i : i + BATCH_SIZE], encoding_method)

        # State preparation
        batch_states = []
        for vec_idx, vec in enumerate(batch):
            qc = QuantumCircuit(n_qubits)
            if encoding_method == "basis":
                idx = int(vec)
                for bit in range(n_qubits):
                    if (idx >> bit) & 1:
                        qc.x(bit)
            elif encoding_method == "angle":
                for qubit, angle in enumerate(vec):
                    qc.ry(2.0 * float(angle), qubit)
            else:
                qc.initialize(vec, range(n_qubits))
            qc.save_statevector()
            t_qc = transpile(qc, backend)
            result = backend.run(t_qc).result().get_statevector().data
            batch_states.append(result)

            if (vec_idx + 1) % 10 == 0:
                print(f"    Processed {vec_idx + 1}/{len(batch)} vectors...", end="\r")

        # Transfer to GPU
        gpu_tensor = torch.tensor(
            np.array(batch_states), device="cuda", dtype=torch.complex64
        )
        all_qiskit_states.append(gpu_tensor)
        _ = model(gpu_tensor.abs())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"\n  Total Time: {total_time:.4f} s")

    all_qiskit_tensor = torch.cat(all_qiskit_states, dim=0)

    # Clean cache after benchmark completion
    clean_cache()

    return total_time, all_qiskit_tensor


# -----------------------------------------------------------
# 2. PennyLane Full Pipeline
# -----------------------------------------------------------
def run_pennylane(n_qubits, n_samples, encoding_method: str = "amplitude"):
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0, None

    # Clean cache before starting benchmark
    clean_cache()

    print(f"\n[PennyLane] Full Pipeline (Disk -> GPU) - {encoding_method} encoding...")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def amplitude_circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    @qml.qnode(dev, interface="torch")
    def basis_circuit(basis_state):
        qml.BasisEmbedding(features=basis_state, wires=range(n_qubits))
        return qml.state()

    @qml.qnode(dev, interface="torch")
    def angle_circuit(inputs):
        qml.AngleEmbedding(features=inputs * 2.0, wires=range(n_qubits), rotation="Y")
        return qml.state()

    model = DummyQNN(n_qubits).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # IO
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    if encoding_method == "basis":
        raw_data = df["index"].values.astype(np.int64)
    else:
        raw_data = np.stack(df["feature_vector"].values)
    io_time = time.perf_counter() - start_time
    print(f"  IO Time: {io_time:.4f} s")

    all_pl_states = []

    # Process batches
    for i in range(0, n_samples, BATCH_SIZE):
        if encoding_method == "basis":
            batch_indices = raw_data[i : i + BATCH_SIZE]
            # Convert indices to binary representation for BasisEmbedding
            batch_states = []
            for idx in batch_indices:
                binary_list = [int(b) for b in format(int(idx), f"0{n_qubits}b")]
                state_cpu = basis_circuit(binary_list)
                batch_states.append(state_cpu)
            state_cpu = torch.stack(batch_states)
        elif encoding_method == "angle":
            batch_cpu = torch.tensor(raw_data[i : i + BATCH_SIZE])
            # Execute QNode
            try:
                state_cpu = angle_circuit(batch_cpu)
            except Exception:
                state_cpu = torch.stack([angle_circuit(x) for x in batch_cpu])
        else:
            batch_cpu = torch.tensor(raw_data[i : i + BATCH_SIZE])
            # Execute QNode
            try:
                state_cpu = amplitude_circuit(batch_cpu)
            except Exception:
                state_cpu = torch.stack([amplitude_circuit(x) for x in batch_cpu])

        all_pl_states.append(state_cpu)

        # Transfer to GPU
        state_gpu = state_cpu.to("cuda", dtype=torch.float32)
        _ = model(state_gpu.abs())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")

    # Stack all collected states
    all_pl_states_tensor = torch.cat(
        all_pl_states, dim=0
    )  # Should handle cases where last batch is smaller

    # Clean cache after benchmark completion
    clean_cache()

    return total_time, all_pl_states_tensor


# -----------------------------------------------------------
# 3. Mahout Parquet Pipeline
# -----------------------------------------------------------
def _load_parquet_iqp_batch(n_qubits: int, encoding_method: str) -> np.ndarray:
    table = pq.read_table(DATA_FILE)
    rows = table.column("feature_vector").to_pylist()
    return np.asarray(rows, dtype=np.float64)


def _load_arrow_iqp_batch(n_qubits: int, encoding_method: str) -> np.ndarray:
    with pa.memory_map(ARROW_FILE, "r") as source:
        reader = ipc.open_file(source)
        table = reader.read_all()
    flat = table.column("data").flatten().to_numpy(zero_copy_only=False)
    dim = _input_dim(n_qubits, encoding_method)
    return flat.reshape(-1, dim)


def run_mahout_parquet(
    engine,
    n_qubits,
    n_samples,
    encoding_method: str = "amplitude",
    encode_path: str = "fwt",
):
    # Clean cache before starting benchmark
    clean_cache()

    path_tag = "TC" if encode_path == "tc" else "FWT"
    print(
        f"\n[Mahout-Parquet/{path_tag}] Full Pipeline (Parquet -> GPU) "
        f"— {encoding_method}, path={encode_path}..."
    )
    model = DummyQNN(n_qubits).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    parquet_encode_start = time.perf_counter()
    if encode_path == "tc":
        batch = _load_parquet_iqp_batch(n_qubits, encoding_method)
        qtensor = _mahout_encode_batch(
            engine, batch, n_qubits, encoding_method, encode_path
        )
    else:
        qtensor = engine.encode(DATA_FILE, n_qubits, encoding_method)
    parquet_encode_time = time.perf_counter() - parquet_encode_start
    print(f"  Parquet->GPU (IO+Encode): {parquet_encode_time:.4f} s")

    # Convert to torch tensor
    dlpack_start = time.perf_counter()
    gpu_batched = torch.from_dlpack(qtensor)
    dlpack_time = time.perf_counter() - dlpack_start
    print(f"  DLPack conversion: {dlpack_time:.4f} s")

    # Tensor is already 2D [n_samples, state_len]
    state_len = 1 << n_qubits
    assert gpu_batched.shape == (n_samples, state_len), (
        f"Expected shape ({n_samples}, {state_len}), got {gpu_batched.shape}"
    )

    # Convert to float for model
    reshape_start = time.perf_counter()
    gpu_all_data = gpu_batched.abs().to(torch.float32)
    reshape_time = time.perf_counter() - reshape_start
    print(f"  Convert to float32: {reshape_time:.4f} s")

    # Forward pass (data already on GPU)
    for i in range(0, n_samples, BATCH_SIZE):
        batch = gpu_all_data[i : i + BATCH_SIZE]
        _ = model(batch)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")

    # Clean cache after benchmark completion
    clean_cache()

    return total_time, gpu_batched


# -----------------------------------------------------------
# 4. Mahout Arrow IPC Pipeline
# -----------------------------------------------------------
def run_mahout_arrow(
    engine,
    n_qubits,
    n_samples,
    encoding_method: str = "amplitude",
    encode_path: str = "fwt",
):
    # Clean cache before starting benchmark
    clean_cache()

    path_tag = "TC" if encode_path == "tc" else "FWT"
    print(
        f"\n[Mahout-Arrow/{path_tag}] Full Pipeline (Arrow IPC -> GPU) "
        f"— {encoding_method}, path={encode_path}..."
    )
    model = DummyQNN(n_qubits).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    arrow_encode_start = time.perf_counter()
    if encode_path == "tc":
        batch = _load_arrow_iqp_batch(n_qubits, encoding_method)
        qtensor = _mahout_encode_batch(
            engine, batch, n_qubits, encoding_method, encode_path
        )
    else:
        qtensor = engine.encode(ARROW_FILE, n_qubits, encoding_method)
    arrow_encode_time = time.perf_counter() - arrow_encode_start
    print(f"  Arrow->GPU (IO+Encode): {arrow_encode_time:.4f} s")

    dlpack_start = time.perf_counter()
    gpu_batched = torch.from_dlpack(qtensor)
    dlpack_time = time.perf_counter() - dlpack_start
    print(f"  DLPack conversion: {dlpack_time:.4f} s")

    # Tensor is already 2D [n_samples, state_len]
    state_len = 1 << n_qubits
    assert gpu_batched.shape == (n_samples, state_len), (
        f"Expected shape ({n_samples}, {state_len}), got {gpu_batched.shape}"
    )

    reshape_start = time.perf_counter()
    gpu_all_data = gpu_batched.abs().to(torch.float32)
    reshape_time = time.perf_counter() - reshape_start
    print(f"  Convert to float32: {reshape_time:.4f} s")

    for i in range(0, n_samples, BATCH_SIZE):
        batch = gpu_all_data[i : i + BATCH_SIZE]
        _ = model(batch)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")

    # Clean cache after benchmark completion
    clean_cache()

    return total_time, gpu_batched


def compare_states(name_a, states_a, name_b, states_b) -> None:
    print("\n" + "=" * 70)
    print(f"VERIFICATION ({name_a} vs {name_b})")
    print("=" * 70)

    # Ensure both tensors are on GPU for comparison
    n_compare = min(len(states_a), len(states_b))
    tensor_a = states_a[:n_compare].cuda()
    tensor_b = states_b[:n_compare].cuda()

    # Compare Probabilities (|psi|^2)
    diff_probs = (tensor_a.abs() ** 2 - tensor_b.abs() ** 2).abs().max().item()
    print(f"Max Probability Difference: {diff_probs:.2e}")

    # Compare Raw Amplitudes
    # We compare full complex difference magnitude
    diff_amps = (tensor_a - tensor_b).abs().max().item()
    print(f"Max Amplitude Difference:   {diff_amps:.2e}")

    if diff_probs < 1e-5:
        print(">> SUCCESS: Quantum States Match!")
    else:
        print(">> FAILURE: States do not match.")


def verify_correctness(states_dict) -> None:
    # Filter out None values
    valid_states = {
        name: states for name, states in states_dict.items() if states is not None
    }

    if len(valid_states) < 2:
        return

    keys = sorted(list(valid_states.keys()))
    for name_a, name_b in itertools.combinations(keys, 2):
        compare_states(name_a, valid_states[name_a], name_b, valid_states[name_b])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final End-to-End Benchmark (Disk -> GPU VRAM)"
    )
    parser.add_argument(
        "--qubits", type=int, default=16, help="Number of qubits (16 recommended)"
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="Number of training samples"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["mahout-parquet", "pennylane"],
        choices=["mahout-parquet", "mahout-arrow", "pennylane", "qiskit", "all"],
        help="Frameworks to benchmark. Use 'all' to run all available frameworks.",
    )
    parser.add_argument(
        "--encoding-method",
        type=str,
        default="amplitude",
        choices=["amplitude", "angle", "basis", "iqp", "iqp-z"],
        help="Encoding method (iqp/iqp-z enable Tensor Core path option).",
    )
    parser.add_argument(
        "--encode-path",
        type=str,
        default="fwt",
        choices=["fwt", "tc", "both"],
        help="Mahout GPU path: fwt=encode(), tc=encode_batch_tc(), both=compare.",
    )
    args = parser.parse_args()

    if args.encode_path in ("tc", "both") and args.encoding_method not in (
        "iqp",
        "iqp-z",
    ):
        parser.error("encode_path tc/both requires --encoding-method iqp or iqp-z")

    # Expand "all" option
    if "all" in args.frameworks:
        args.frameworks = ["mahout-parquet", "mahout-arrow", "pennylane", "qiskit"]

    generate_data(args.qubits, args.samples, args.encoding_method)

    try:
        engine = QdpEngine(0)
    except Exception as e:
        print(f"Mahout Init Error: {e}")
        exit(1)

    # Clean cache before starting benchmarks
    clean_cache()

    print("\n" + "=" * 70)
    print(f"E2E BENCHMARK: {args.qubits} Qubits, {args.samples} Samples")
    print("=" * 70)

    timing_results: dict[str, float] = {}
    state_results: dict[str, object] = {}

    if "pennylane" in args.frameworks:
        t_pl, pl_all_states = run_pennylane(
            args.qubits, args.samples, args.encoding_method
        )
        if t_pl > 0:
            timing_results["PennyLane"] = t_pl
            state_results["PennyLane"] = pl_all_states
        clean_cache()

    if "qiskit" in args.frameworks:
        t_qiskit, qiskit_all_states = run_qiskit(
            args.qubits, args.samples, args.encoding_method
        )
        if t_qiskit > 0:
            timing_results["Qiskit"] = t_qiskit
            state_results["Qiskit"] = qiskit_all_states
        clean_cache()

    encode_paths = ["fwt", "tc"] if args.encode_path == "both" else [args.encode_path]

    for encode_path in encode_paths:
        path_suffix = f"-{encode_path.upper()}" if args.encode_path == "both" else ""

        if "mahout-parquet" in args.frameworks:
            t, states = run_mahout_parquet(
                engine,
                args.qubits,
                args.samples,
                args.encoding_method,
                encode_path=encode_path,
            )
            if t > 0:
                timing_results[f"Mahout-Parquet{path_suffix}"] = t
                state_results[f"Mahout-Parquet{path_suffix}"] = states
            clean_cache()

        if "mahout-arrow" in args.frameworks:
            t, states = run_mahout_arrow(
                engine,
                args.qubits,
                args.samples,
                args.encoding_method,
                encode_path=encode_path,
            )
            if t > 0:
                timing_results[f"Mahout-Arrow{path_suffix}"] = t
                state_results[f"Mahout-Arrow{path_suffix}"] = states
            clean_cache()

    print("\n" + "=" * 70)
    print("E2E LATENCY (Lower is Better)")
    print(
        f"Samples: {args.samples}, Qubits: {args.qubits}, "
        f"encoding={args.encoding_method}, encode_path={args.encode_path}"
    )
    print("=" * 70)

    sorted_results = sorted(timing_results.items(), key=lambda x: x[1])
    for name, time_val in sorted_results:
        print(f"{name:24s} {time_val:10.4f} s")

    if args.encode_path == "both":
        for base in ("Mahout-Parquet", "Mahout-Arrow"):
            fwt_key, tc_key = f"{base}-FWT", f"{base}-TC"
            if fwt_key in timing_results and tc_key in timing_results:
                ratio = timing_results[fwt_key] / timing_results[tc_key]
                print(f"{base} FWT/TC speedup: {ratio:.2f}x")

    print("-" * 70)
    mahout_times = [
        t for name, t in timing_results.items() if name.startswith("Mahout-") and t > 0
    ]
    t_mahout_best = min(mahout_times) if mahout_times else 0
    if t_mahout_best > 0:
        if "PennyLane" in timing_results:
            print(
                f"Speedup vs PennyLane: "
                f"{timing_results['PennyLane'] / t_mahout_best:10.2f}x"
            )
        if "Qiskit" in timing_results:
            print(
                f"Speedup vs Qiskit:    "
                f"{timing_results['Qiskit'] / t_mahout_best:10.2f}x"
            )

    if args.encoding_method not in ("iqp", "iqp-z"):
        verify_correctness(state_results)
