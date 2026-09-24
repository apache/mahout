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

"""IQP shared-memory FWT fusion benchmark (PR3).

For num_qubits <= FWT_SHARED_MEM_THRESHOLD the batch path fuses phase
computation, shared-memory FWT, and normalization in one kernel per sample.

Run from repo root::

    uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_pr3.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from qumat_qdp import QdpEngine

FWT_SHARED_MEM_THRESHOLD = 12


def benchmark_iqp_batch(
    num_qubits: int,
    num_samples: int,
    iters: int = 50,
) -> float:
    """Return average batch IQP encode latency in microseconds."""
    data_len = num_qubits + num_qubits * (num_qubits - 1) // 2
    batch_data = np.random.randn(num_samples, data_len).astype(np.float64)
    engine = QdpEngine(0)

    for _ in range(5):
        _ = engine.encode(batch_data, num_qubits, "iqp")
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = engine.encode(batch_data, num_qubits, "iqp")
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) / iters * 1e6
    return elapsed_us


def main() -> None:
    parser = argparse.ArgumentParser(description="IQP shared-memory FWT benchmark")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[8, 10, 12, 13, 14],
        help="Qubit counts to benchmark",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Cannot benchmark.")

    device_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print("IQP shared-memory FWT fusion benchmark (PR3)")
    print(f"GPU: {device_name}")
    print(
        f"Config: batch_size={args.batch_size}, iterations={args.iterations}, "
        f"threshold={FWT_SHARED_MEM_THRESHOLD}"
    )
    print("=" * 70)
    print(f"{'Qubits':<8} {'Path':<22} {'Time (us)':>12}")
    print("-" * 70)

    for n in args.qubits:
        path = "shared-mem fused" if n <= FWT_SHARED_MEM_THRESHOLD else "global-mem FWT"
        latency_us = benchmark_iqp_batch(n, args.batch_size, args.iterations)
        print(f"{n:<8} {path:<22} {latency_us:>12.1f}")


if __name__ == "__main__":
    main()
