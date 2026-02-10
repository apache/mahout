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
Benchmark API: Rust-optimized pipeline only (no Python for loop).

Usage:
    from qumat_qdp import QdpBenchmark, ThroughputResult, LatencyResult

    result = (QdpBenchmark(device_id=0).qubits(16).encoding("amplitude")
              .batches(100, size=64).warmup(2).run_throughput())
    # result.duration_sec, result.vectors_per_sec

    lat = (QdpBenchmark(device_id=0).qubits(16).encoding("amplitude")
           .batches(100, size=64).run_latency())
    # lat.latency_ms_per_vector
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
_run_throughput_pipeline_py: Optional[object] = None


def _get_run_throughput_pipeline_py():
    """Return Rust run_throughput_pipeline_py; raise if not available."""
    global _run_throughput_pipeline_py
    if _run_throughput_pipeline_py is not None:
        return _run_throughput_pipeline_py
    import _qdp

    fn = getattr(_qdp, "run_throughput_pipeline_py", None)
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
    Builder for throughput/latency benchmarks. Backend is Rust optimized pipeline only.

    No Python for loop; run_throughput_pipeline_py runs the full pipeline in Rust
    (single Python boundary, GIL released). Requires _qdp.run_throughput_pipeline_py
    (Linux/CUDA build).
    """

    def __init__(self, device_id: int = 0):
        self._device_id = device_id
        self._num_qubits: Optional[int] = None
        self._encoding_method: str = "amplitude"
        self._total_batches: Optional[int] = None
        self._batch_size: int = 64
        self._warmup_batches: int = 0

    def qubits(self, n: int) -> "QdpBenchmark":
        self._num_qubits = n
        return self

    def encoding(self, method: str) -> "QdpBenchmark":
        self._encoding_method = method
        return self

    def batches(self, total: int, size: int = 64) -> "QdpBenchmark":
        self._total_batches = total
        self._batch_size = size
        return self

    def prefetch(self, n: int) -> "QdpBenchmark":
        """No-op for API compatibility; Rust pipeline does not use prefetch from Python."""
        return self

    def warmup(self, n: int) -> "QdpBenchmark":
        self._warmup_batches = n
        return self

    def run_throughput(self) -> ThroughputResult:
        """Run throughput via Rust optimized pipeline (no Python for loop)."""
        if self._num_qubits is None or self._total_batches is None:
            raise ValueError(
                "Set qubits and batches (e.g. .qubits(16).batches(100, 64))"
            )

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

    def run_latency(self) -> LatencyResult:
        """Run latency via Rust optimized pipeline (no Python for loop)."""
        if self._num_qubits is None or self._total_batches is None:
            raise ValueError(
                "Set qubits and batches (e.g. .qubits(16).batches(100, 64))"
            )

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
