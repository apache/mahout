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
PyTorch Tensor latency benchmark: Tests PyTorch tensor zero-copy optimization.

This is a modified version of benchmark_latency.py that uses PyTorch tensors
instead of NumPy arrays to test the zero-copy optimization.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from _qdp import QdpEngine
from utils import build_sample, normalize_batch_torch, prefetched_batches_torch

BAR = "=" * 70
SEP = "-" * 70


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_mahout_pytorch(
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    prefetch: int,
    encoding_method: str = "amplitude",
):
    """Run Mahout benchmark with PyTorch tensor input."""
    try:
        engine = QdpEngine(0)
    except Exception as exc:
        print(f"[Mahout-PyTorch] Init failed: {exc}")
        return 0.0, 0.0

    vector_len = num_qubits if encoding_method == "angle" else (1 << num_qubits)
    sync_cuda()
    start = time.perf_counter()
    processed = 0

    for batch_tensor in prefetched_batches_torch(
        total_batches, batch_size, vector_len, prefetch, encoding_method
    ):
        normalized = normalize_batch_torch(batch_tensor, encoding_method)
        qtensor = engine.encode(normalized, num_qubits, encoding_method)
        _ = torch.utils.dlpack.from_dlpack(qtensor)
        processed += normalized.shape[0]

    sync_cuda()
    duration = time.perf_counter() - start
    latency_ms = (duration / processed) * 1000 if processed > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({latency_ms:.3f} ms/vector)")
    return duration, latency_ms


def run_mahout_numpy(
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    prefetch: int,
    encoding_method: str = "amplitude",
):
    """Run Mahout benchmark with NumPy array input (for comparison)."""
    try:
        engine = QdpEngine(0)
    except Exception as exc:
        print(f"[Mahout-NumPy] Init failed: {exc}")
        return 0.0, 0.0

    vector_len = num_qubits if encoding_method == "angle" else (1 << num_qubits)
    sync_cuda()
    start = time.perf_counter()
    processed = 0

    # Use the same data generation but keep as NumPy
    for batch_idx in range(total_batches):
        base = batch_idx * batch_size
        batch = [
            build_sample(base + i, vector_len, encoding_method)
            for i in range(batch_size)
        ]
        batch_np = np.stack(batch)
        if encoding_method in ("basis", "angle"):
            normalized = batch_np
        else:
            norms = np.linalg.norm(batch_np, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = batch_np / norms

        qtensor = engine.encode(normalized, num_qubits, encoding_method)
        _ = torch.utils.dlpack.from_dlpack(qtensor)
        processed += normalized.shape[0]

    sync_cuda()
    duration = time.perf_counter() - start
    latency_ms = (duration / processed) * 1000 if processed > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({latency_ms:.3f} ms/vector)")
    return duration, latency_ms


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch Tensor encoding latency (zero-copy optimization test)."
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=16,
        help="Number of qubits (power-of-two vector length).",
    )
    parser.add_argument("--batches", type=int, default=100, help="Total batches.")
    parser.add_argument("--batch-size", type=int, default=32, help="Vectors per batch.")
    parser.add_argument(
        "--prefetch", type=int, default=16, help="CPU-side prefetch depth."
    )
    parser.add_argument(
        "--encoding-method",
        type=str,
        default="amplitude",
        choices=["amplitude", "angle", "basis"],
        help="Encoding method to use for Mahout (amplitude, angle, or basis).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device not available; GPU is required.")

    total_vectors = args.batches * args.batch_size
    vector_len = 1 << args.qubits

    print(
        f"PyTorch Tensor Encoding Benchmark: {args.qubits} Qubits, {total_vectors} Samples"
    )
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Vector length: {vector_len}")
    print(f"  Batches      : {args.batches}")
    print(f"  Prefetch     : {args.prefetch}")
    print()

    print(BAR)
    print(
        f"PYTORCH TENSOR LATENCY BENCHMARK: {args.qubits} Qubits, {total_vectors} Samples"
    )
    print(BAR)

    print()
    print("[Mahout-PyTorch] PyTorch Tensor Input (Zero-Copy Optimization)...")
    t_pytorch, l_pytorch = run_mahout_pytorch(
        args.qubits, args.batches, args.batch_size, args.prefetch, args.encoding_method
    )

    print()
    print("[Mahout-NumPy] NumPy Array Input (Baseline)...")
    t_numpy, l_numpy = run_mahout_numpy(
        args.qubits, args.batches, args.batch_size, args.prefetch, args.encoding_method
    )

    print()
    print(BAR)
    print("LATENCY COMPARISON (Lower is Better)")
    print(f"Samples: {total_vectors}, Qubits: {args.qubits}")
    print(BAR)
    print(f"{'PyTorch Tensor':18s} {l_pytorch:10.3f} ms/vector")
    print(f"{'NumPy Array':18s} {l_numpy:10.3f} ms/vector")

    if l_numpy > 0 and l_pytorch > 0:
        print(SEP)
        speedup = l_numpy / l_pytorch
        improvement = ((l_numpy - l_pytorch) / l_numpy) * 100
        print(f"Speedup: {speedup:.2f}x")
        print(f"Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
