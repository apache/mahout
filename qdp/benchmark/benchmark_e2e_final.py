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

import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import itertools
import pyarrow as pa
import pyarrow.parquet as pq
from mahout_qdp import QdpEngine

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
HIDDEN_DIM = 16
BATCH_SIZE = 64  # Small batch to stress loop overhead


class DummyQNN(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.fc = nn.Linear(1 << n_qubits, HIDDEN_DIM)

    def forward(self, x):
        return self.fc(x)


def generate_data(n_qubits, n_samples):
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

    print(f"Generating {n_samples} samples of {n_qubits} qubits...")
    dim = 1 << n_qubits

    # Generate for PennyLane/Qiskit (List format)
    chunk_size = 500
    schema_list = pa.schema([("feature_vector", pa.list_(pa.float64()))])

    with pq.ParquetWriter(DATA_FILE, schema_list) as writer:
        for start_idx in range(0, n_samples, chunk_size):
            current = min(chunk_size, n_samples - start_idx)
            data = np.random.rand(current, dim).astype(np.float64)
            feature_vectors = [row.tolist() for row in data]
            arrays = pa.array(feature_vectors, type=pa.list_(pa.float64()))
            batch_table = pa.Table.from_arrays([arrays], names=["feature_vector"])
            writer.write_table(batch_table)

    file_size_mb = os.path.getsize(DATA_FILE) / (1024 * 1024)
    print(f"  Generated {n_samples} samples")
    print(f"  Parquet file size: {file_size_mb:.2f} MB")


# -----------------------------------------------------------
# 1. Qiskit Full Pipeline
# -----------------------------------------------------------
def run_qiskit(n_qubits, n_samples):
    if not HAS_QISKIT:
        print("\n[Qiskit] Not installed, skipping.")
        return 0.0, None

    print("\n[Qiskit] Full Pipeline (Disk -> GPU)...")
    model = DummyQNN(n_qubits).cuda()
    backend = AerSimulator(method="statevector")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # IO
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data = np.stack(df["feature_vector"].values)
    io_time = time.perf_counter() - start_time
    print(f"  IO Time: {io_time:.4f} s")

    all_qiskit_states = []

    # Process batches
    for i in range(0, n_samples, BATCH_SIZE):
        batch = raw_data[i : i + BATCH_SIZE]

        # Normalize
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        batch = batch / norms

        # State preparation
        batch_states = []
        for vec_idx, vec in enumerate(batch):
            qc = QuantumCircuit(n_qubits)
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
    return total_time, all_qiskit_tensor


# -----------------------------------------------------------
# 2. PennyLane Full Pipeline
# -----------------------------------------------------------
def run_pennylane(n_qubits, n_samples):
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0, None

    print("\n[PennyLane] Full Pipeline (Disk -> GPU)...")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    model = DummyQNN(n_qubits).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # IO
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data = np.stack(df["feature_vector"].values)
    io_time = time.perf_counter() - start_time
    print(f"  IO Time: {io_time:.4f} s")

    all_pl_states = []

    # Process batches
    for i in range(0, n_samples, BATCH_SIZE):
        batch_cpu = torch.tensor(raw_data[i : i + BATCH_SIZE])

        # Execute QNode
        try:
            state_cpu = circuit(batch_cpu)
        except Exception:
            state_cpu = torch.stack([circuit(x) for x in batch_cpu])

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

    return total_time, all_pl_states_tensor


# -----------------------------------------------------------
# 3. Mahout Full Pipeline
# -----------------------------------------------------------
def run_mahout(engine, n_qubits, n_samples):
    print("\n[Mahout] Full Pipeline (Disk -> GPU)...")
    model = DummyQNN(n_qubits).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Direct Parquet to GPU pipeline
    parquet_encode_start = time.perf_counter()
    batched_tensor = engine.encode_from_parquet(DATA_FILE, n_qubits, "amplitude")
    parquet_encode_time = time.perf_counter() - parquet_encode_start
    print(f"  Parquet->GPU (IO+Encode): {parquet_encode_time:.4f} s")

    # Convert to torch tensor (single DLPack call)
    dlpack_start = time.perf_counter()
    gpu_batched = torch.from_dlpack(batched_tensor)
    dlpack_time = time.perf_counter() - dlpack_start
    print(f"  DLPack conversion: {dlpack_time:.4f} s")

    # Reshape to [n_samples, state_len] (still complex)
    state_len = 1 << n_qubits
    gpu_reshaped = gpu_batched.view(n_samples, state_len)

    # Convert to float for model (batch already on GPU)
    reshape_start = time.perf_counter()
    gpu_all_data = gpu_reshaped.abs().to(torch.float32)
    reshape_time = time.perf_counter() - reshape_start
    print(f"  Reshape & convert: {reshape_time:.4f} s")

    # Forward pass (data already on GPU)
    for i in range(0, n_samples, BATCH_SIZE):
        batch = gpu_all_data[i : i + BATCH_SIZE]
        _ = model(batch)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")
    return total_time, gpu_reshaped


def compare_states(name_a, states_a, name_b, states_b):
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


def verify_correctness(states_dict):
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
        default=["mahout", "pennylane"],
        choices=["mahout", "pennylane", "qiskit", "all"],
        help="Frameworks to benchmark (default: mahout pennylane). Use 'all' to run all available frameworks.",
    )
    args = parser.parse_args()

    # Expand "all" option
    if "all" in args.frameworks:
        all_frameworks = []
        if "mahout" in args.frameworks or "all" in args.frameworks:
            all_frameworks.append("mahout")
        if "pennylane" in args.frameworks or "all" in args.frameworks:
            all_frameworks.append("pennylane")
        if "qiskit" in args.frameworks or "all" in args.frameworks:
            all_frameworks.append("qiskit")
        args.frameworks = all_frameworks

    generate_data(args.qubits, args.samples)

    try:
        engine = QdpEngine(0)
    except Exception as e:
        print(f"Mahout Init Error: {e}")
        exit(1)

    print("\n" + "=" * 70)
    print(f"E2E BENCHMARK: {args.qubits} Qubits, {args.samples} Samples")
    print("=" * 70)

    # Initialize results
    t_pl, pl_all_states = 0.0, None
    t_mahout, mahout_all_states = 0.0, None
    t_qiskit, qiskit_all_states = 0.0, None

    # Run benchmarks
    if "pennylane" in args.frameworks:
        t_pl, pl_all_states = run_pennylane(args.qubits, args.samples)

    if "qiskit" in args.frameworks:
        t_qiskit, qiskit_all_states = run_qiskit(args.qubits, args.samples)

    if "mahout" in args.frameworks:
        t_mahout, mahout_all_states = run_mahout(engine, args.qubits, args.samples)

    print("\n" + "=" * 70)
    print("E2E LATENCY (Lower is Better)")
    print(f"Samples: {args.samples}, Qubits: {args.qubits}")
    print("=" * 70)

    results = []
    if t_mahout > 0:
        results.append(("Mahout", t_mahout))
    if t_pl > 0:
        results.append(("PennyLane", t_pl))
    if t_qiskit > 0:
        results.append(("Qiskit", t_qiskit))

    results.sort(key=lambda x: x[1])

    for name, time_val in results:
        print(f"{name:12s} {time_val:10.4f} s")

    print("-" * 70)
    if t_mahout > 0:
        if t_pl > 0:
            print(f"Speedup vs PennyLane: {t_pl / t_mahout:10.2f}x")
        if t_qiskit > 0:
            print(f"Speedup vs Qiskit:    {t_qiskit / t_mahout:10.2f}x")

    # Run Verification after benchmarks
    verify_correctness(
        {
            "Mahout": mahout_all_states,
            "PennyLane": pl_all_states,
            "Qiskit": qiskit_all_states,
        }
    )
