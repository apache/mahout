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
Throughput benchmark using QuantumDataLoader (for qt in loader).

Compares iterator-based throughput with run_throughput_pipeline_py.
Expectation: loader version slightly slower due to Python boundary per batch.

Usage (from qdp-python):
  uv run python benchmark/benchmark_loader_throughput.py --qubits 16 --batches 200 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so qumat_qdp is importable when run as script
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qumat_qdp import QuantumDataLoader, QdpBenchmark  # noqa: E402


def run_loader_throughput(
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    encoding_method: str = "amplitude",
) -> tuple[float, float]:
    """Run throughput by iterating QuantumDataLoader; returns (duration_sec, vectors_per_sec)."""
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(num_qubits)
        .encoding(encoding_method)
        .batches(total_batches, size=batch_size)
        .source_synthetic()
    )
    total_vectors = total_batches * batch_size
    start = time.perf_counter()
    count = 0
    for qt in loader:
        count += 1
        # Consumer: touch tensor (e.g. could torch.from_dlpack(qt) and use it)
        _ = qt
    elapsed = time.perf_counter() - start
    if count != total_batches:
        raise RuntimeError(f"Expected {total_batches} batches, got {count}")
    duration_sec = max(elapsed, 1e-9)
    vectors_per_sec = total_vectors / duration_sec
    return duration_sec, vectors_per_sec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuantumDataLoader throughput benchmark"
    )
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoding", type=str, default="amplitude")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    print("QuantumDataLoader throughput (for qt in loader)")
    print(
        f"  qubits={args.qubits}, batches={args.batches}, batch_size={args.batch_size}"
    )

    loader_times: list[float] = []
    for t in range(args.trials):
        dur, vps = run_loader_throughput(
            args.qubits, args.batches, args.batch_size, args.encoding
        )
        loader_times.append(vps)
        print(f"  Trial {t + 1}: {dur:.4f} s, {vps:.1f} vec/s")

    median_vps = sorted(loader_times)[len(loader_times) // 2]
    print(f"  Median: {median_vps:.1f} vec/s")

    # Compare with full Rust pipeline (single boundary)
    print("\nQdpBenchmark.run_throughput() (full Rust pipeline, single boundary):")
    result = (
        QdpBenchmark(device_id=0)
        .qubits(args.qubits)
        .encoding(args.encoding)
        .batches(args.batches, size=args.batch_size)
        .run_throughput()
    )
    print(f"  {result.duration_sec:.4f} s, {result.vectors_per_sec:.1f} vec/s")
    print(f"\nLoader vs pipeline ratio: {median_vps / result.vectors_per_sec:.3f}")


if __name__ == "__main__":
    main()
