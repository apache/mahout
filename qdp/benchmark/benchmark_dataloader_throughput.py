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
DataLoader throughput benchmark across Mahout (QDP), PennyLane, and Qiskit.

The workload mirrors the `qdp-core/examples/dataloader_throughput.rs` pipeline:
- Generate batches of size `BATCH_SIZE` with deterministic vectors.
- Prefetch on the CPU side to keep the GPU fed.
- Encode vectors into amplitude states on GPU and run a tiny consumer op.

Run:
    python qdp/benchmark/benchmark_dataloader_throughput.py --qubits 16 --batches 200 --batch-size 64
"""

import argparse
import queue
import threading
import time

import numpy as np
import torch

from mahout_qdp import QdpEngine

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


def build_sample(seed: int, vector_len: int) -> np.ndarray:
    mask = np.uint64(vector_len - 1)
    scale = 1.0 / vector_len
    idx = np.arange(vector_len, dtype=np.uint64)
    mixed = (idx + np.uint64(seed)) & mask
    return mixed.astype(np.float64) * scale


def prefetched_batches(total_batches: int, batch_size: int, vector_len: int, prefetch: int):
    q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=prefetch)

    def producer():
        for batch_idx in range(total_batches):
            base = batch_idx * batch_size
            batch = [build_sample(base + i, vector_len) for i in range(batch_size)]
            q.put(np.stack(batch))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch


def normalize_batch(batch: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return batch / norms


def run_mahout(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    try:
        engine = QdpEngine(0)
    except Exception as exc:
        print(f"[Mahout] Init failed: {exc}")
        return 0.0, 0.0

    total_vectors = total_batches * batch_size
    torch.cuda.synchronize()
    start = time.perf_counter()

    processed = 0
    for batch in prefetched_batches(total_batches, batch_size, 1 << num_qubits, prefetch):
        normalized = normalize_batch(batch)
        for sample in normalized:
            qtensor = engine.encode(sample.tolist(), num_qubits, "amplitude")
            tensor = torch.utils.dlpack.from_dlpack(qtensor).abs().to(torch.float32)
            _ = tensor.sum()
            processed += 1

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  IO + Encode Time: {duration:.4f} s")
    print(f"  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_pennylane(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    if not HAS_PENNYLANE:
        print("[PennyLane] Not installed, skipping.")
        return 0.0, 0.0

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    torch.cuda.synchronize()
    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(total_batches, batch_size, 1 << num_qubits, prefetch):
        batch_cpu = torch.tensor(batch, dtype=torch.float64)
        try:
            state_cpu = circuit(batch_cpu)
        except Exception:
            state_cpu = torch.stack([circuit(x) for x in batch_cpu])
        state_gpu = state_cpu.to("cuda", dtype=torch.float32)
        _ = state_gpu.abs().sum()
        processed += len(batch_cpu)

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_qiskit(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    if not HAS_QISKIT:
        print("[Qiskit] Not installed, skipping.")
        return 0.0, 0.0

    backend = AerSimulator(method="statevector")
    torch.cuda.synchronize()
    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(total_batches, batch_size, 1 << num_qubits, prefetch):
        normalized = normalize_batch(batch)

        batch_states = []
        for vec_idx, vec in enumerate(normalized):
            qc = QuantumCircuit(num_qubits)
            qc.initialize(vec, range(num_qubits))
            qc.save_statevector()
            t_qc = transpile(qc, backend)
            state = backend.run(t_qc).result().get_statevector().data
            batch_states.append(state)
            processed += 1
            if (vec_idx + 1) % 10 == 0:
                print(f"    Processed {vec_idx + 1}/{len(normalized)} vectors...", end="\r")

        gpu_tensor = torch.tensor(np.array(batch_states), device="cuda", dtype=torch.complex64)
        _ = gpu_tensor.abs().sum()

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"\n  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def main():
    parser = argparse.ArgumentParser(description="Benchmark DataLoader throughput across frameworks.")
    parser.add_argument("--qubits", type=int, default=16, help="Number of qubits (power-of-two vector length).")
    parser.add_argument("--batches", type=int, default=200, help="Total batches to stream.")
    parser.add_argument("--batch-size", type=int, default=64, help="Vectors per batch.")
    parser.add_argument("--prefetch", type=int, default=16, help="CPU-side prefetch depth.")
    args = parser.parse_args()

    total_vectors = args.batches * args.batch_size
    vector_len = 1 << args.qubits

    print(f"Generating {total_vectors} samples of {args.qubits} qubits...")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Vector length: {vector_len}")
    print(f"  Batches      : {args.batches}")
    print(f"  Prefetch     : {args.prefetch}")
    bytes_per_vec = vector_len * 8
    print(f"  Generated {total_vectors} samples")
    print(f"  PennyLane/Qiskit format: {total_vectors * bytes_per_vec / (1024 * 1024):.2f} MB")
    print(f"  Mahout format: {total_vectors * bytes_per_vec / (1024 * 1024):.2f} MB")
    print()

    print("=" * 70)
    print(f"E2E BENCHMARK: {args.qubits} Qubits, {total_vectors} Samples")
    print("=" * 70)

    print("\n[PennyLane] Full Pipeline (DataLoader -> GPU)...")
    t_pl, th_pl = run_pennylane(args.qubits, args.batches, args.batch_size, args.prefetch)

    print("\n[Qiskit] Full Pipeline (DataLoader -> GPU)...")
    t_qiskit, th_qiskit = run_qiskit(args.qubits, args.batches, args.batch_size, args.prefetch)

    print("\n[Mahout] Full Pipeline (DataLoader -> GPU)...")
    t_mahout, th_mahout = run_mahout(args.qubits, args.batches, args.batch_size, args.prefetch)

    print("\n" + "=" * 70)
    print("E2E LATENCY (Lower is Better)")
    print(f"Samples: {total_vectors}, Qubits: {args.qubits}")
    print("=" * 70)

    latency_results = []
    throughput_results = []
    if t_pl > 0:
        latency_results.append(("PennyLane", t_pl))
        throughput_results.append(("PennyLane", th_pl))
    if t_qiskit > 0:
        latency_results.append(("Qiskit", t_qiskit))
        throughput_results.append(("Qiskit", th_qiskit))
    if t_mahout > 0:
        latency_results.append(("Mahout", t_mahout))
        throughput_results.append(("Mahout", th_mahout))

    latency_results.sort(key=lambda x: x[1])
    throughput_results.sort(key=lambda x: x[1], reverse=True)

    for name, tval in latency_results:
        print(f"{name:12s} {tval:10.4f} s")

    print("-" * 70)
    for name, tput in throughput_results:
        print(f"{name:12s} {tput:10.1f} vectors/sec")

    if t_mahout > 0:
        print("-" * 70)
        if t_pl > 0:
            print(f"Speedup vs PennyLane: {t_pl / t_mahout:10.2f}x")
        if t_qiskit > 0:
            print(f"Speedup vs Qiskit:    {t_qiskit / t_mahout:10.2f}x")


if __name__ == "__main__":
    main()
