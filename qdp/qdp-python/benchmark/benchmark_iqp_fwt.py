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
IQP Encoding FWT Optimization Benchmark

Measures performance improvement from Fast Walsh-Hadamard Transform optimization
which reduces complexity from O(4^n) to O(n * 2^n).

Run:
    python qdp/qdp-python/benchmark/benchmark_iqp_fwt.py
    python qdp/qdp-python/benchmark/benchmark_iqp_fwt.py --qubits 4 6 8 10 12
    python qdp/qdp-python/benchmark/benchmark_iqp_fwt.py --batch-size 64 --iterations 100
"""

from __future__ import annotations

import argparse
import time

import torch
import numpy as np

from _qdp import QdpEngine

BAR = "=" * 80
SEP = "-" * 80


def sync_cuda() -> None:
    """Synchronize CUDA device for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def iqp_z_data_len(num_qubits: int) -> int:
    """Calculate data length for IQP-Z encoding (n parameters)."""
    return num_qubits


def iqp_full_data_len(num_qubits: int) -> int:
    """Calculate data length for full IQP encoding (n + n*(n-1)/2 parameters)."""
    return num_qubits + num_qubits * (num_qubits - 1) // 2


def theoretical_speedup(n: int) -> float:
    """Calculate theoretical speedup from FWT optimization.

    Old: O(4^n) = O(2^(2n))
    New: O(n * 2^n)
    Speedup: 2^n / n
    """
    return (2 ** n) / n


def benchmark_single_encode(
    engine: QdpEngine,
    num_qubits: int,
    encoding: str,
    iterations: int,
    warmup: int = 5,
) -> tuple[float, float]:
    """Benchmark single sample IQP encoding.

    Returns:
        (avg_time_ms, throughput_samples_per_sec)
    """
    data_len = iqp_full_data_len(num_qubits) if encoding == "iqp" else iqp_z_data_len(num_qubits)
    data = [0.1 * i for i in range(data_len)]

    # Warmup
    for _ in range(warmup):
        qtensor = engine.encode(data, num_qubits, encoding)
        _ = torch.from_dlpack(qtensor)
    sync_cuda()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        qtensor = engine.encode(data, num_qubits, encoding)
        _ = torch.from_dlpack(qtensor)
    sync_cuda()
    duration = time.perf_counter() - start

    avg_time_ms = (duration / iterations) * 1000
    throughput = iterations / duration

    return avg_time_ms, throughput


def benchmark_batch_encode(
    engine: QdpEngine,
    num_qubits: int,
    encoding: str,
    batch_size: int,
    iterations: int,
    warmup: int = 5,
) -> tuple[float, float]:
    """Benchmark batch IQP encoding.

    Returns:
        (avg_time_ms, throughput_samples_per_sec)
    """
    data_len = iqp_full_data_len(num_qubits) if encoding == "iqp" else iqp_z_data_len(num_qubits)

    # Create batch data
    batch_data = torch.tensor(
        [[0.1 * (i + j * data_len) for i in range(data_len)] for j in range(batch_size)],
        dtype=torch.float64
    )

    # Warmup
    for _ in range(warmup):
        qtensor = engine.encode(batch_data, num_qubits, encoding)
        _ = torch.from_dlpack(qtensor)
    sync_cuda()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        qtensor = engine.encode(batch_data, num_qubits, encoding)
        _ = torch.from_dlpack(qtensor)
    sync_cuda()
    duration = time.perf_counter() - start

    avg_time_ms = (duration / iterations) * 1000
    total_samples = iterations * batch_size
    throughput = total_samples / duration

    return avg_time_ms, throughput


def verify_correctness(engine: QdpEngine, num_qubits: int, encoding: str) -> bool:
    """Verify that encoding produces normalized quantum states."""
    data_len = iqp_full_data_len(num_qubits) if encoding == "iqp" else iqp_z_data_len(num_qubits)
    data = [0.1 * i for i in range(data_len)]

    qtensor = engine.encode(data, num_qubits, encoding)
    torch_tensor = torch.from_dlpack(qtensor)

    # Check normalization
    norm = torch.sum(torch.abs(torch_tensor) ** 2).item()
    return abs(norm - 1.0) < 1e-6


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark IQP encoding FWT optimization performance."
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12],
        help="Qubit counts to benchmark (default: 4 6 8 10 12).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for batch encoding benchmark (default: 64).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations per benchmark (default: 50).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["iqp", "iqp-z", "both"],
        default="both",
        help="Encoding type to benchmark (default: both).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify correctness before benchmarking.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device not available; GPU is required.")

    try:
        engine = QdpEngine(0)
    except Exception as exc:
        raise SystemExit(f"Failed to initialize QdpEngine: {exc}")

    encodings = ["iqp", "iqp-z"] if args.encoding == "both" else [args.encoding]

    print(BAR)
    print("IQP ENCODING FWT OPTIMIZATION BENCHMARK")
    print(BAR)
    print(f"Qubit counts: {args.qubits}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Encodings: {', '.join(encodings)}")
    print()

    # Verify correctness if requested
    if args.verify:
        print("Verifying correctness...")
        all_correct = True
        for enc in encodings:
            for n in args.qubits:
                correct = verify_correctness(engine, n, enc)
                status = "PASS" if correct else "FAIL"
                print(f"  {enc} n={n}: {status}")
                all_correct = all_correct and correct
        if not all_correct:
            raise SystemExit("Correctness verification failed!")
        print("All correctness checks passed.\n")

    # Single sample benchmark
    print(BAR)
    print("SINGLE SAMPLE ENCODING PERFORMANCE")
    print(BAR)
    print(f"{'Encoding':<10} {'Qubits':>6} {'Time (ms)':>12} {'Throughput':>15} {'Theory Speedup':>15}")
    print(SEP)

    for enc in encodings:
        for n in args.qubits:
            avg_time, throughput = benchmark_single_encode(
                engine, n, enc, args.iterations, args.warmup
            )
            theory = theoretical_speedup(n)
            print(f"{enc:<10} {n:>6} {avg_time:>12.4f} {throughput:>12.1f}/s {theory:>15.1f}x")
        print()

    # Batch encoding benchmark
    print(BAR)
    print(f"BATCH ENCODING PERFORMANCE (batch_size={args.batch_size})")
    print(BAR)
    print(f"{'Encoding':<10} {'Qubits':>6} {'Batch Time':>12} {'Throughput':>15} {'State Size':>12}")
    print(SEP)

    for enc in encodings:
        for n in args.qubits:
            avg_time, throughput = benchmark_batch_encode(
                engine, n, enc, args.batch_size, args.iterations, args.warmup
            )
            state_size = 2 ** n
            print(f"{enc:<10} {n:>6} {avg_time:>10.4f}ms {throughput:>12.1f}/s {state_size:>12}")
        print()

    # Summary with theoretical vs actual analysis
    print(BAR)
    print("THEORETICAL COMPLEXITY ANALYSIS")
    print(BAR)
    print("FWT optimization reduces complexity from O(4^n) to O(n * 2^n)")
    print()
    print(f"{'Qubits':>6} {'Old O(4^n)':>15} {'New O(n*2^n)':>15} {'Theory Speedup':>15}")
    print(SEP)

    for n in args.qubits:
        old_ops = 4 ** n
        new_ops = n * (2 ** n)
        speedup = old_ops / new_ops
        print(f"{n:>6} {old_ops:>15,} {new_ops:>15,} {speedup:>15.1f}x")

    print()
    print("Note: Actual speedup may differ due to memory bandwidth, kernel launch")
    print("overhead, and GPU utilization factors.")


if __name__ == "__main__":
    main()
