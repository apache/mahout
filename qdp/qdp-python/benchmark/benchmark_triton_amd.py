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

"""Baseline benchmark for Triton AMD backend (ROCm)."""

from __future__ import annotations

import argparse

import torch
from qumat_qdp import TritonAmdKernel, is_triton_amd_available


def _build_input(method: str, batch_size: int, qubits: int) -> torch.Tensor:
    if method == "basis":
        return torch.randint(
            low=0,
            high=1 << qubits,
            size=(batch_size,),
            device="cuda",
            dtype=torch.int64,
        )
    if method == "angle":
        return torch.randn(batch_size, qubits, device="cuda", dtype=torch.float32)
    return torch.randn(batch_size, 1 << qubits, device="cuda", dtype=torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton AMD backend throughput/latency."
    )
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument(
        "--encoding-method",
        type=str,
        default="amplitude",
        choices=["amplitude", "angle", "basis"],
    )
    parser.add_argument(
        "--precision", type=str, default="float32", choices=["float32", "float64"]
    )
    args = parser.parse_args()

    if not is_triton_amd_available():
        raise SystemExit(
            "triton_amd backend is unavailable (requires ROCm + Triton HIP target)."
        )

    engine = TritonAmdKernel(device_id=0, precision=args.precision)
    data = _build_input(args.encoding_method, args.batch_size, args.qubits)

    # Warmup
    for _ in range(10):
        _ = engine.encode(data, args.qubits, args.encoding_method)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    processed = 0
    start.record()
    for _ in range(args.batches):
        _ = engine.encode(data, args.qubits, args.encoding_method)
        processed += args.batch_size
    end.record()
    torch.cuda.synchronize()
    dt = start.elapsed_time(end) / 1000.0

    throughput = processed / dt if dt > 0 else 0.0
    latency_ms_per_vector = (dt / processed) * 1000 if processed else 0.0

    print("TRITON AMD BASELINE")
    print(f"- Encoding: {args.encoding_method}")
    print(f"- Qubits: {args.qubits}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Batches: {args.batches}")
    print(f"- Duration: {dt:.4f} s")
    print(f"- Throughput: {throughput:.1f} vectors/sec")
    print(f"- Latency: {latency_ms_per_vector:.6f} ms/vector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
