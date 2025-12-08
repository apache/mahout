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

    MAHOUT_DATA_FILE = DATA_FILE.replace(".parquet", "_mahout.parquet")
    if os.path.exists(MAHOUT_DATA_FILE):
        os.remove(MAHOUT_DATA_FILE)

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

    # Generate for Mahout (flat Float64 format, one sample per batch)
    schema_flat = pa.schema([("data", pa.float64())])
    with pq.ParquetWriter(MAHOUT_DATA_FILE, schema_flat) as writer:
        for i in range(n_samples):
            sample_data = np.random.rand(dim).astype(np.float64)
            array = pa.array(sample_data, type=pa.float64())
            batch_table = pa.Table.from_arrays([array], names=["data"])
            writer.write_table(batch_table)

    file_size_mb = os.path.getsize(DATA_FILE) / (1024 * 1024)
    mahout_size_mb = os.path.getsize(MAHOUT_DATA_FILE) / (1024 * 1024)
    print(f"  Generated {n_samples} samples")
    print(f"  PennyLane/Qiskit format: {file_size_mb:.2f} MB")
    print(f"  Mahout format: {mahout_size_mb:.2f} MB")


# -----------------------------------------------------------
# 1. Qiskit Full Pipeline
# -----------------------------------------------------------
def run_qiskit(n_qubits, n_samples):
    if not HAS_QISKIT:
        print("\n[Qiskit] Not installed, skipping.")
        return 0.0

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
        _ = model(gpu_tensor.abs())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"\n  Total Time: {total_time:.4f} s")
    return total_time


# -----------------------------------------------------------
# 2. PennyLane Full Pipeline
# -----------------------------------------------------------
def run_pennylane(n_qubits, n_samples):
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0

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

    # Process batches
    for i in range(0, n_samples, BATCH_SIZE):
        batch_cpu = torch.tensor(raw_data[i : i + BATCH_SIZE])

        # Execute QNode
        try:
            state_cpu = circuit(batch_cpu)
        except Exception:
            state_cpu = torch.stack([circuit(x) for x in batch_cpu])

        # Transfer to GPU
        state_gpu = state_cpu.to("cuda", dtype=torch.float32)
        _ = model(state_gpu.abs())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")
    return total_time


# -----------------------------------------------------------
# 3. Mahout Full Pipeline
# -----------------------------------------------------------
def run_mahout(engine, n_qubits, n_samples):
    print("\n[Mahout] Full Pipeline (Disk -> GPU)...")
    model = DummyQNN(n_qubits).cuda()
    MAHOUT_DATA_FILE = DATA_FILE.replace(".parquet", "_mahout.parquet")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Read Parquet and encode all samples
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(MAHOUT_DATA_FILE)

    all_states = []
    for batch in parquet_file.iter_batches():
        sample_data = batch.column(0).to_numpy()
        qtensor = engine.encode(sample_data.tolist(), n_qubits, "amplitude")
        gpu_state = torch.from_dlpack(qtensor)
        all_states.append(gpu_state)

    # Stack and convert
    gpu_all_data = torch.stack(all_states).abs().to(torch.float32)

    encode_time = time.perf_counter() - start_time
    print(f"  IO + Encode Time: {encode_time:.4f} s")

    # Forward pass (data already on GPU)
    for i in range(0, n_samples, BATCH_SIZE):
        batch = gpu_all_data[i : i + BATCH_SIZE]
        _ = model(batch)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")
    return total_time


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
    args = parser.parse_args()

    generate_data(args.qubits, args.samples)

    try:
        engine = QdpEngine(0)
    except Exception as e:
        print(f"Mahout Init Error: {e}")
        exit(1)

    print("\n" + "=" * 70)
    print(f"E2E BENCHMARK: {args.qubits} Qubits, {args.samples} Samples")
    print("=" * 70)

    # Run benchmarks
    t_pl = run_pennylane(args.qubits, args.samples)
    t_qiskit = run_qiskit(args.qubits, args.samples)
    t_mahout = run_mahout(engine, args.qubits, args.samples)

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
