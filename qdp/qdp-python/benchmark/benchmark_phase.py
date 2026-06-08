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

"""GPU phase encoding latency benchmark.

Measures batch phase encoding on the current build. For before/after comparisons,
checkout the baseline commit, rebuild the extension, run with ``--label baseline``,
then repeat on the optimized branch with ``--label optimized``.

Run from repo root::

    uv run --project qdp/qdp-python python qdp/qdp-python/benchmark/benchmark_phase.py
"""

from __future__ import annotations

import argparse

import torch
from qumat_qdp import QdpEngine


def benchmark_phase(
    num_qubits: int,
    num_samples: int,
    iters: int = 5,
) -> tuple[float, float]:
    """Return (total_us, per_sample_us) for batch phase encoding on GPU."""
    phases = torch.tensor(
        [[0.1 * (k + 1) for k in range(num_qubits)]] * num_samples,
        dtype=torch.float64,
        device="cuda",
    )
    engine = QdpEngine(0)

    for _ in range(3):
        qtensor = engine.encode(phases, num_qubits, "phase")
        _ = torch.from_dlpack(qtensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        qtensor = engine.encode(phases, num_qubits, "phase")
        _ = torch.from_dlpack(qtensor)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end) / iters
    total_us = total_ms * 1000.0
    per_sample_us = total_us / num_samples
    return total_us, per_sample_us


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU phase encoding benchmark")
    parser.add_argument("--qubits", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--label",
        choices=("baseline", "optimized"),
        default="optimized",
        help="Tag for the current checkout (baseline or optimized)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Cannot benchmark.")

    device_name = torch.cuda.get_device_name(0)
    total_us, per_sample_us = benchmark_phase(
        args.qubits, args.batch_size, args.iterations
    )

    print("=" * 70)
    print("Phase encoding GPU benchmark")
    print(f"GPU: {device_name}")
    print(
        f"Config: qubits={args.qubits}, batch_size={args.batch_size}, "
        f"iterations={args.iterations}, label={args.label}"
    )
    print("=" * 70)
    print(f"Total batch time:  {total_us:.1f} us")
    print(f"Per sample:        {per_sample_us:.2f} us ({per_sample_us / 1000:.3f} ms)")


if __name__ == "__main__":
    main()
