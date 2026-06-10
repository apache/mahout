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

"""IQP Ozaki implicit Hadamard benchmark (PR5).

GPU-vs-GPU timing only (no PyTorch reference). Compare paths via ``--path``:

- ``fwt``  — ``engine.encode(..., "iqp")`` (standard FWT dispatch)
- ``tc``   — ``engine.encode_batch_tc(...)`` (Ozaki Kronecker TC path)
- ``both`` — run both and print speedup (FWT / TC)

Before/after PR5: checkout ``pr4-kronecker-fwt`` and run ``--path tc`` (naive
GEMM scaffold), then ``pr5-implicit-hadamard-engine`` with ``--path tc`` (Ozaki).

Run from repo root::

    python qdp/qdp-python/benchmark/benchmark_pr5.py --path both --qubits 14
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from qumat_qdp import QdpEngine

PATH_LABELS = {
    "fwt": "IQP FWT (encode)",
    "tc": "IQP Ozaki TC (encode_batch_tc)",
}


def _iqp_param_count(num_qubits: int) -> int:
    return num_qubits + num_qubits * (num_qubits - 1) // 2


def benchmark_path(
    engine: QdpEngine,
    path: str,
    num_qubits: int,
    num_samples: int,
    iters: int,
) -> float:
    """Return average batch latency in milliseconds."""
    data_len = _iqp_param_count(num_qubits)
    batch_data = np.random.randn(num_samples, data_len).astype(np.float64)

    for _ in range(5):
        if path == "fwt":
            _ = engine.encode(batch_data, num_qubits, "iqp")
        else:
            _ = engine.encode_batch_tc(batch_data, num_qubits)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        if path == "fwt":
            _ = engine.encode(batch_data, num_qubits, "iqp")
        else:
            _ = engine.encode_batch_tc(batch_data, num_qubits)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="IQP Ozaki TC benchmark (PR5)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--path",
        choices=["fwt", "tc", "both"],
        default="both",
        help="Encoding path to benchmark (GPU vs GPU)",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[12, 14, 16],
        help="Qubit counts to benchmark",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional run label (e.g. before-pr5, after-pr5)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Cannot benchmark.")

    device_name = torch.cuda.get_device_name(0)
    print("=" * 72)
    print("IQP Ozaki implicit Hadamard benchmark (PR5)")
    if args.label:
        print(f"Label: {args.label}")
    print(f"GPU: {device_name}")
    print(
        f"Config: batch_size={args.batch_size}, iterations={args.iterations}, "
        f"path={args.path}"
    )
    print("=" * 72)

    engine = QdpEngine(0, precision="float64")
    if args.path in ("tc", "both") and not hasattr(engine, "encode_batch_tc"):
        raise SystemExit("encode_batch_tc not available in this build.")

    if args.path == "both":
        print(
            f"{'Qubits':<8} {'Dim':<10} {'FWT (ms)':>12} {'TC (ms)':>12} "
            f"{'Speedup':>10}"
        )
        print("-" * 72)
        for n in args.qubits:
            dim = 1 << n
            fwt_ms = benchmark_path(engine, "fwt", n, args.batch_size, args.iterations)
            tc_ms = benchmark_path(engine, "tc", n, args.batch_size, args.iterations)
            speedup = fwt_ms / tc_ms if tc_ms > 0 else float("inf")
            print(f"{n:<8} {dim:<10} {fwt_ms:>12.3f} {tc_ms:>12.3f} {speedup:>9.2f}x")
    else:
        label = PATH_LABELS[args.path]
        print(f"{'Qubits':<8} {'Dim':<10} {'Path':<32} {'Time (ms)':>12}")
        print("-" * 72)
        for n in args.qubits:
            dim = 1 << n
            latency_ms = benchmark_path(
                engine, args.path, n, args.batch_size, args.iterations
            )
            print(f"{n:<8} {dim:<10} {label:<32} {latency_ms:>12.3f}")


if __name__ == "__main__":
    main()
