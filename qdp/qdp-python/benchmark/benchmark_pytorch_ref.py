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
Benchmark: QDP Rust+CUDA vs PyTorch reference implementation.

Compares encoding throughput in two modes:

* **encode-only** (default): Data is pre-generated on the target device
  before the timer starts.  Both PyTorch and Mahout time encoding kernels
  only, giving the fairest kernel-vs-kernel comparison.

* **end-to-end**: Data generation, CPU→GPU transfer, and encoding are all
  inside the timer.  Matches the full pipeline cost users actually pay.

Usage:
    # Fair kernel comparison (default):
    python benchmark_pytorch_ref.py --qubits 16 --batches 100 --batch-size 64

    # Full pipeline comparison:
    python benchmark_pytorch_ref.py --mode end-to-end --qubits 16 --batches 100

    # Specific frameworks:
    python benchmark_pytorch_ref.py --encoding-method angle --frameworks pytorch-gpu,mahout
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from utils import generate_batch_data

FRAMEWORK_CHOICES = ("pytorch-cpu", "pytorch-gpu", "mahout")


def _parse_frameworks(raw: str) -> list[str]:
    """Parse comma-separated framework list; ``'all'`` expands to all choices."""
    if raw.strip().lower() == "all":
        return list(FRAMEWORK_CHOICES)
    names = [s.strip() for s in raw.split(",")]
    for n in names:
        if n not in FRAMEWORK_CHOICES:
            raise argparse.ArgumentTypeError(
                f"Unknown framework {n!r}. Choose from: {', '.join(FRAMEWORK_CHOICES)}"
            )
    return names


def _sample_dim(encoding_method: str, num_qubits: int) -> int:
    if encoding_method == "basis":
        return 1
    if encoding_method == "angle":
        return num_qubits
    if encoding_method == "iqp":
        return num_qubits + num_qubits * (num_qubits - 1) // 2
    return 1 << num_qubits


def _generate_batches(
    total: int,
    batch_size: int,
    sample_dim: int,
    encoding_method: str,
    device: str,
) -> list[torch.Tensor]:
    """Pre-generate batch data using the shared utility (deterministic)."""
    batches = []
    for b in range(total):
        np_data = generate_batch_data(
            batch_size,
            sample_dim,
            encoding_method,
            seed=42 + b,
        )
        if encoding_method == "basis":
            t = torch.tensor(np_data.flatten(), dtype=torch.float64, device=device)
        else:
            t = torch.tensor(np_data, dtype=torch.float64, device=device)
        batches.append(t)
    return batches


def run_pytorch(
    *,
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    encoding_method: str,
    warmup_batches: int,
    device: str,
) -> tuple[float, float]:
    """Run PyTorch reference encoding and return (duration_sec, vectors_per_sec)."""
    from qumat_qdp.torch_ref import encode

    dim = _sample_dim(encoding_method, num_qubits)
    all_batches = _generate_batches(
        warmup_batches + total_batches,
        batch_size,
        dim,
        encoding_method,
        device,
    )

    # Warmup.
    for b in range(warmup_batches):
        encode(all_batches[b], num_qubits, encoding_method, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timed run.
    start = time.perf_counter()
    for b in range(warmup_batches, len(all_batches)):
        encode(all_batches[b], num_qubits, encoding_method, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    duration = time.perf_counter() - start

    total_vectors = total_batches * batch_size
    vps = total_vectors / duration if duration > 0 else 0.0
    return duration, vps


def run_mahout(
    *,
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    encoding_method: str,
    warmup_batches: int,
    device_id: int,
) -> tuple[float, float]:
    """Run QDP Rust+CUDA pipeline and return (duration_sec, vectors_per_sec)."""
    from qumat_qdp.api import QdpBenchmark

    result = (
        QdpBenchmark(device_id=device_id)
        .backend("rust")
        .qubits(num_qubits)
        .encoding(encoding_method)
        .batches(total_batches, size=batch_size)
        .warmup(warmup_batches)
        .run_throughput()
    )
    return result.duration_sec, result.vectors_per_sec


def run_mahout_encode_only(
    *,
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    encoding_method: str,
    warmup_batches: int,
    device_id: int,
) -> tuple[float, float]:
    """Run Mahout encoding from pre-generated CUDA tensors (encode-only)."""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id)
    device = f"cuda:{device_id}"
    dim = _sample_dim(encoding_method, num_qubits)
    all_batches = _generate_batches(
        warmup_batches + total_batches,
        batch_size,
        dim,
        encoding_method,
        device,
    )

    # Rust basis encoding expects int64 CUDA tensors.
    if encoding_method == "basis":
        all_batches = [b.to(torch.int64) for b in all_batches]

    # Warmup.
    for b in range(warmup_batches):
        qt = engine.encode(all_batches[b], num_qubits, encoding_method)
        _ = torch.utils.dlpack.from_dlpack(qt)
    torch.cuda.synchronize()

    # Timed run.
    start = time.perf_counter()
    for b in range(warmup_batches, len(all_batches)):
        qt = engine.encode(all_batches[b], num_qubits, encoding_method)
        _ = torch.utils.dlpack.from_dlpack(qt)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start

    total_vectors = total_batches * batch_size
    return duration, total_vectors / duration if duration > 0 else 0.0


def run_pytorch_end_to_end(
    *,
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    encoding_method: str,
    warmup_batches: int,
    device: str,
) -> tuple[float, float]:
    """Run PyTorch encoding with data generation inside the timer (end-to-end)."""
    from qumat_qdp.torch_ref import encode

    dim = _sample_dim(encoding_method, num_qubits)

    # Warmup (data gen outside timer is OK for warmup).
    for b in range(warmup_batches):
        np_data = generate_batch_data(batch_size, dim, encoding_method, seed=42 + b)
        t = torch.tensor(
            np_data.flatten() if encoding_method == "basis" else np_data,
            dtype=torch.float64,
            device=device,
        )
        encode(t, num_qubits, encoding_method, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timed run: data gen + transfer + encode.
    start = time.perf_counter()
    for b in range(total_batches):
        np_data = generate_batch_data(
            batch_size,
            dim,
            encoding_method,
            seed=42 + warmup_batches + b,
        )
        t = torch.tensor(
            np_data.flatten() if encoding_method == "basis" else np_data,
            dtype=torch.float64,
            device=device,
        )
        encode(t, num_qubits, encoding_method, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    duration = time.perf_counter() - start

    total_vectors = total_batches * batch_size
    return duration, total_vectors / duration if duration > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="QDP vs PyTorch reference benchmark")
    parser.add_argument(
        "--mode",
        default="encode-only",
        choices=["encode-only", "end-to-end"],
        help="'encode-only': data pre-generated on GPU, times encoding only. "
        "'end-to-end': data generation + transfer + encoding all timed.",
    )
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--encoding-method",
        default="amplitude",
        choices=["amplitude", "angle", "basis", "iqp"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--frameworks",
        type=_parse_frameworks,
        default=list(FRAMEWORK_CHOICES),
    )
    args = parser.parse_args()

    enc = args.encoding_method
    mode = args.mode

    print(f"\n{'=' * 60}")
    print(f"QDP vs PyTorch Reference Benchmark ({mode})")
    print(f"{'=' * 60}")
    print(f"  Mode:        {mode}")
    print(f"  Qubits:      {args.qubits}")
    print(f"  Batches:     {args.batches}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Encoding:    {enc}")
    print(f"  Warmup:      {args.warmup}")
    print(f"  Trials:      {args.trials}")
    print(f"  Total vecs:  {args.batches * args.batch_size:,}")
    print(f"{'=' * 60}")
    if mode == "encode-only":
        print("  Note: 'encode-only' times encoding kernels only;")
        print("        data is pre-generated on the target device.")
    else:
        print("  Note: 'end-to-end' times data generation + transfer + encoding.")
    print()

    results: dict[str, float] = {}  # framework -> median vps

    for fw in args.frameworks:
        trial_vps: list[float] = []
        try:
            for trial in range(args.trials):
                if fw == "pytorch-cpu":
                    pytorch_fn = (
                        run_pytorch if mode == "encode-only" else run_pytorch_end_to_end
                    )
                    dur, vps = pytorch_fn(
                        num_qubits=args.qubits,
                        total_batches=args.batches,
                        batch_size=args.batch_size,
                        encoding_method=enc,
                        warmup_batches=args.warmup,
                        device="cpu",
                    )
                elif fw == "pytorch-gpu":
                    if not torch.cuda.is_available():
                        print(f"  {fw:20s}  SKIPPED (no CUDA)")
                        break
                    pytorch_fn = (
                        run_pytorch if mode == "encode-only" else run_pytorch_end_to_end
                    )
                    dur, vps = pytorch_fn(
                        num_qubits=args.qubits,
                        total_batches=args.batches,
                        batch_size=args.batch_size,
                        encoding_method=enc,
                        warmup_batches=args.warmup,
                        device=f"cuda:{args.device_id}",
                    )
                elif fw == "mahout":
                    if not torch.cuda.is_available():
                        print(f"  {fw:20s}  SKIPPED (no CUDA)")
                        break
                    if mode == "encode-only":
                        dur, vps = run_mahout_encode_only(
                            num_qubits=args.qubits,
                            total_batches=args.batches,
                            batch_size=args.batch_size,
                            encoding_method=enc,
                            warmup_batches=args.warmup,
                            device_id=args.device_id,
                        )
                    else:
                        dur, vps = run_mahout(
                            num_qubits=args.qubits,
                            total_batches=args.batches,
                            batch_size=args.batch_size,
                            encoding_method=enc,
                            warmup_batches=args.warmup,
                            device_id=args.device_id,
                        )
                else:
                    continue
                trial_vps.append(vps)

            if trial_vps:
                median = statistics.median(trial_vps)
                results[fw] = median
                print(
                    f"  {fw:20s}  {median:>12,.0f} vec/s  (median of {len(trial_vps)} trials)"
                )
        except Exception as e:
            print(f"  {fw:20s}  ERROR: {e}")

    # Speedup ratios.
    if len(results) > 1:
        print(f"\n{'Speedup Ratios':^60}")
        print(f"{'-' * 60}")
        baselines = [k for k in ("pytorch-gpu", "pytorch-cpu") if k in results]
        for base in baselines:
            base_vps = results[base]
            for fw in results:
                if fw != base and fw not in baselines and base_vps > 0:
                    ratio = results[fw] / base_vps
                    print(f"  {fw} vs {base}: {ratio:.1f}x")

    print()


if __name__ == "__main__":
    main()
