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
Benchmark API: supports Rust-optimized pipeline and PyTorch reference backend.

Usage:
    from qumat_qdp import QdpBenchmark, ThroughputResult, LatencyResult

    # Rust backend (default):
    result = (QdpBenchmark(device_id=0).qubits(16).encoding("amplitude")
              .batches(100, size=64).warmup(2).run_throughput())

    # PyTorch reference (must be explicitly selected):
    result = (QdpBenchmark(device_id=0).backend("pytorch").qubits(16)
              .encoding("amplitude").batches(100, size=64).run_throughput())
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ThroughputResult:
    """Result of run_throughput(): duration and vectors per second."""

    duration_sec: float
    vectors_per_sec: float


@dataclass
class LatencyResult:
    """Result of run_latency(): duration and ms per vector."""

    duration_sec: float
    latency_ms_per_vector: float


# Cached reference to Rust pipeline (avoids repeated import).
_run_throughput_pipeline_py: object | None = None


def _get_run_throughput_pipeline_py():
    """Return Rust run_throughput_pipeline_py; raise if not available."""
    global _run_throughput_pipeline_py
    if _run_throughput_pipeline_py is not None:
        return _run_throughput_pipeline_py

    from qumat_qdp._backend import get_qdp

    qdp = get_qdp()
    if qdp is None:
        raise RuntimeError(
            "Rust pipeline not available: _qdp extension not found. "
            "Build the extension with: maturin develop"
        )

    fn = getattr(qdp, "run_throughput_pipeline_py", None)
    if fn is None:
        raise RuntimeError(
            "Rust pipeline not available: _qdp.run_throughput_pipeline_py is missing. "
            "Force uv to rebuild the extension: from qdp-python run `uv sync --refresh-package qumat-qdp` "
            "then `uv run python benchmark/run_pipeline_baseline.py`. Or run `maturin develop` and use "
            "`.venv/bin/python` or `benchmark/run_baseline.sh`."
        )
    _run_throughput_pipeline_py = fn
    return fn


class QdpBenchmark:
    """
    Builder for throughput/latency benchmarks.

    Supports two backends:
    - ``"rust"`` (default): Rust-optimized pipeline (no Python for-loop, GIL released).
    - ``"pytorch"``: Pure PyTorch reference implementation (must be explicitly selected).
    """

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id
        self._num_qubits: int | None = None
        self._encoding_method: str = "amplitude"
        self._total_batches: int | None = None
        self._batch_size: int = 64
        self._warmup_batches: int = 0
        self._backend_name: str = "rust"

    def qubits(self, n: int) -> QdpBenchmark:
        self._num_qubits = n
        return self

    def encoding(self, method: str) -> QdpBenchmark:
        self._encoding_method = method
        return self

    def batches(self, total: int, size: int = 64) -> QdpBenchmark:
        self._total_batches = total
        self._batch_size = size
        return self

    def prefetch(self, n: int) -> QdpBenchmark:
        """No-op for API compatibility; Rust pipeline does not use prefetch from Python."""
        return self

    def warmup(self, n: int) -> QdpBenchmark:
        self._warmup_batches = n
        return self

    def backend(self, name: str) -> QdpBenchmark:
        """Set benchmark backend: ``'rust'`` or ``'pytorch'``."""
        if name not in ("rust", "pytorch"):
            raise ValueError(
                f"backend must be 'rust' or 'pytorch', got {name!r}"
            )
        self._backend_name = name
        return self

    def _validate(self) -> None:
        if self._num_qubits is None or self._total_batches is None:
            raise ValueError(
                "Set qubits and batches (e.g. .qubits(16).batches(100, 64))"
            )

    def run_throughput(self) -> ThroughputResult:
        """Run throughput benchmark using the selected backend."""
        self._validate()
        if self._backend_name == "pytorch":
            return self._run_throughput_pytorch()
        return self._run_throughput_rust()

    def run_latency(self) -> LatencyResult:
        """Run latency benchmark using the selected backend."""
        self._validate()
        if self._backend_name == "pytorch":
            return self._run_latency_pytorch()
        return self._run_latency_rust()

    # -- Rust backend --

    def _run_throughput_rust(self) -> ThroughputResult:
        run_rust = _get_run_throughput_pipeline_py()
        duration_sec, vectors_per_sec, _ = run_rust(
            device_id=self._device_id,
            num_qubits=self._num_qubits,
            batch_size=self._batch_size,
            total_batches=self._total_batches,
            encoding_method=self._encoding_method,
            warmup_batches=self._warmup_batches,
            seed=None,
        )
        return ThroughputResult(
            duration_sec=duration_sec, vectors_per_sec=vectors_per_sec
        )

    def _run_latency_rust(self) -> LatencyResult:
        run_rust = _get_run_throughput_pipeline_py()
        duration_sec, _, latency_ms_per_vector = run_rust(
            device_id=self._device_id,
            num_qubits=self._num_qubits,
            batch_size=self._batch_size,
            total_batches=self._total_batches,
            encoding_method=self._encoding_method,
            warmup_batches=self._warmup_batches,
            seed=None,
        )
        return LatencyResult(
            duration_sec=duration_sec,
            latency_ms_per_vector=latency_ms_per_vector,
        )

    # -- PyTorch backend --

    def _run_throughput_pytorch(self) -> ThroughputResult:
        import torch

        from qumat_qdp.torch_ref import encode

        device = f"cuda:{self._device_id}" if torch.cuda.is_available() else "cpu"
        # _validate() guarantees these are not None.
        assert self._num_qubits is not None
        assert self._total_batches is not None
        num_qubits = self._num_qubits
        encoding_method = self._encoding_method
        batch_size = self._batch_size

        if encoding_method == "basis":
            sample_dim = 1
        elif encoding_method == "angle":
            sample_dim = num_qubits
        elif encoding_method == "iqp":
            sample_dim = num_qubits + num_qubits * (num_qubits - 1) // 2
        else:
            sample_dim = 1 << num_qubits

        # Generate all batch data upfront.
        batches = []
        for b in range(self._total_batches + self._warmup_batches):
            if encoding_method == "basis":
                data = torch.randint(
                    0, 1 << num_qubits, (batch_size,), device=device
                ).to(torch.float64)
            else:
                data = torch.randn(
                    batch_size, sample_dim, dtype=torch.float64, device=device
                )
            batches.append(data)

        # Warmup.
        for b in range(self._warmup_batches):
            encode(batches[b], num_qubits, encoding_method, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        # Timed run.
        start = time.perf_counter()
        for b in range(self._warmup_batches, len(batches)):
            encode(batches[b], num_qubits, encoding_method, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        duration = time.perf_counter() - start

        total_vectors = self._total_batches * batch_size
        return ThroughputResult(
            duration_sec=duration,
            vectors_per_sec=total_vectors / duration if duration > 0 else 0.0,
        )

    def _run_latency_pytorch(self) -> LatencyResult:
        result = self._run_throughput_pytorch()
        assert self._total_batches is not None
        total_vectors = self._total_batches * self._batch_size
        ms_per_vector = (
            (result.duration_sec * 1000.0) / total_vectors if total_vectors > 0 else 0.0
        )
        return LatencyResult(
            duration_sec=result.duration_sec,
            latency_ms_per_vector=ms_per_vector,
        )
