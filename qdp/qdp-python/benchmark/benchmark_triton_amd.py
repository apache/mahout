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
import time

import torch

from qumat_qdp import is_triton_amd_available
from qumat_qdp.triton_amd import TritonAmdEngine


def _build_input(
    batch_size: int,
    num_qubits: int,
    encoding_method: str,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if encoding_method == "basis":
        return torch.randint(0, 1 << num_qubits, (batch_size,), device=device)
    width = num_qubits if encoding_method == "angle" else 1 << num_qubits
    return torch.randn(batch_size, width, device=device, dtype=dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-qubits", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--encoding",
        choices=("amplitude", "angle", "basis"),
        default="amplitude",
    )
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    if not is_triton_amd_available():
        raise SystemExit("Triton AMD backend unavailable.")

    dtype = torch.float32 if args.precision == "float32" else torch.float64
    device = f"cuda:{args.device_id}"
    engine = TritonAmdEngine(device_id=args.device_id, precision=args.precision)
    data = _build_input(
        args.batch_size,
        args.num_qubits,
        args.encoding,
        device=device,
        dtype=dtype,
    )

    for _ in range(5):
        engine.encode(data, args.num_qubits, args.encoding)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        engine.encode(data, args.num_qubits, args.encoding)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start

    total_vectors = args.batch_size * args.iters
    print("Triton AMD benchmark")
    print(f"  encoding:       {args.encoding}")
    print(f"  precision:      {args.precision}")
    print(f"  num_qubits:     {args.num_qubits}")
    print(f"  batch_size:     {args.batch_size}")
    print(f"  iterations:     {args.iters}")
    print(f"  duration_sec:   {duration:.6f}")
    print(f"  vectors_per_s:  {total_vectors / duration:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
