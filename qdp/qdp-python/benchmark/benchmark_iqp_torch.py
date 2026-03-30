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

"""IQP benchmark against a torch reference and ``torch.compile``."""

from __future__ import annotations

import argparse
import time

import torch
from qumat_qdp import QdpEngine

from benchmark.iqp_reference import (
    IQP_ENCODING_METHODS,
    build_iqp_reference,
    iqp_enable_zz,
    iqp_sample_size_for_method,
)
from benchmark.utils import prefetched_batches_torch

BAR = "=" * 70


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _runner_name(kind: str, compiled: bool) -> str:
    if kind == "qdp":
        return "QDP CUDA"
    if compiled:
        return "Torch Reference (compiled)"
    return "Torch Reference (eager)"


def _run_qdp(
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    prefetch: int,
    encoding_method: str,
) -> tuple[float, float]:
    try:
        engine = QdpEngine(0, precision="float64")
    except Exception as exc:
        print(f"[QDP] Init failed: {exc}")
        return 0.0, 0.0

    sample_size = iqp_sample_size_for_method(num_qubits, encoding_method)
    batch_iter = prefetched_batches_torch(
        total_batches,
        batch_size,
        sample_size,
        prefetch,
        encoding_method=encoding_method,
    )
    warmup_batch_cpu = next(batch_iter, None)
    if warmup_batch_cpu is None:
        return 0.0, 0.0

    warmup_batch_gpu = warmup_batch_cpu.to("cuda", non_blocking=True)
    _ = torch.from_dlpack(engine.encode(warmup_batch_gpu, num_qubits, encoding_method))

    sync_cuda()
    start = time.perf_counter()
    processed = 0

    for batch_cpu in batch_iter:
        batch_gpu = batch_cpu.to("cuda", non_blocking=True)
        qtensor = engine.encode(batch_gpu, num_qubits, encoding_method)
        _ = torch.from_dlpack(qtensor)
        processed += batch_gpu.shape[0]

    sync_cuda()
    duration = time.perf_counter() - start
    latency_ms = (duration / processed) * 1000 if processed > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({latency_ms:.3f} ms/vector)")
    return duration, latency_ms


def _run_torch_reference(
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    prefetch: int,
    encoding_method: str,
    *,
    compiled: bool,
) -> tuple[float, float]:
    enable_zz = iqp_enable_zz(encoding_method)
    sample_size = iqp_sample_size_for_method(num_qubits, encoding_method)
    reference = build_iqp_reference(
        num_qubits,
        enable_zz=enable_zz,
        device="cuda",
        dtype=torch.float64,
        compile_reference=compiled,
    )

    batch_iter = prefetched_batches_torch(
        total_batches,
        batch_size,
        sample_size,
        prefetch,
        encoding_method=encoding_method,
    )
    warmup_batch_cpu = next(batch_iter, None)
    if warmup_batch_cpu is None:
        return 0.0, 0.0

    warmup_batch_gpu = warmup_batch_cpu.to("cuda", non_blocking=True)
    _ = reference(warmup_batch_gpu)

    sync_cuda()
    start = time.perf_counter()
    processed = 0

    for batch_idx, batch_cpu in enumerate(batch_iter):
        batch_gpu = batch_cpu.to("cuda", non_blocking=True)
        result = reference(batch_gpu)
        if batch_idx == 0:
            # Sanity-check the reference on the first batch without turning the
            # benchmark into a full correctness test.
            assert result.shape[1] == 1 << num_qubits
        processed += batch_gpu.shape[0]

    sync_cuda()
    duration = time.perf_counter() - start
    latency_ms = (duration / processed) * 1000 if processed > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({latency_ms:.3f} ms/vector)")
    return duration, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark IQP encoding against a torch reference."
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=6,
        help="Number of qubits for the IQP state.",
    )
    parser.add_argument("--batches", type=int, default=100, help="Total batches.")
    parser.add_argument("--batch-size", type=int, default=32, help="Vectors per batch.")
    parser.add_argument(
        "--prefetch", type=int, default=16, help="CPU-side prefetch depth."
    )
    parser.add_argument(
        "--encoding-method",
        type=str,
        default="iqp",
        choices=list(IQP_ENCODING_METHODS),
        help="IQP variant to benchmark.",
    )
    parser.add_argument(
        "--skip-compiled",
        action="store_true",
        help="Skip the torch.compile reference and only run eager + QDP.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device not available; GPU is required.")
    if args.batches < 2:
        raise SystemExit("Use --batches >= 2 so one batch can be reserved for warmup.")

    sample_size = iqp_sample_size_for_method(args.qubits, args.encoding_method)
    timed_batches = args.batches - 1
    total_vectors = timed_batches * args.batch_size

    print(
        f"IQP Benchmark: {args.qubits} qubits, {total_vectors} samples, "
        f"sample_size={sample_size}"
    )
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Batches      : {args.batches}")
    print(f"  Prefetch     : {args.prefetch}")
    print(f"  Encoding     : {args.encoding_method}")
    print()

    print(BAR)
    print("IQP KERNEL BENCHMARK")
    print(BAR)

    print()
    print("[QDP] CUDA kernel...")
    _qdp_duration, qdp_latency = _run_qdp(
        args.qubits, args.batches, args.batch_size, args.prefetch, args.encoding_method
    )

    print()
    print("[Torch] Eager reference...")
    _eager_duration, eager_latency = _run_torch_reference(
        args.qubits,
        args.batches,
        args.batch_size,
        args.prefetch,
        args.encoding_method,
        compiled=False,
    )

    compiled_latency = 0.0
    if not args.skip_compiled:
        print()
        print("[Torch] torch.compile reference...")
        _compiled_duration, compiled_latency = _run_torch_reference(
            args.qubits,
            args.batches,
            args.batch_size,
            args.prefetch,
            args.encoding_method,
            compiled=True,
        )

    print()
    print(BAR)
    print("LATENCY COMPARISON (Lower is Better)")
    print(f"Samples: {total_vectors}, Qubits: {args.qubits} (warmup excluded)")
    print(BAR)
    print(f"{_runner_name('qdp', False):24s} {qdp_latency:10.3f} ms/vector")
    print(f"{_runner_name('torch', False):24s} {eager_latency:10.3f} ms/vector")
    if not args.skip_compiled:
        print(f"{_runner_name('torch', True):24s} {compiled_latency:10.3f} ms/vector")

    if qdp_latency > 0 and eager_latency > 0:
        print("-" * 70)
        print(f"QDP vs eager speedup: {eager_latency / qdp_latency:.2f}x")
        if not args.skip_compiled and compiled_latency > 0:
            print(f"QDP vs compiled speedup: {compiled_latency / qdp_latency:.2f}x")


if __name__ == "__main__":
    main()
