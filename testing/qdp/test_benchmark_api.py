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

"""Tests for the benchmark API (QdpBenchmark, Rust pipeline only)."""

from pathlib import Path

import pytest

# Allow importing benchmark API from qdp-python/benchmark
_bench_dir = (
    Path(__file__).resolve().parent.parent.parent / "qdp" / "qdp-python" / "benchmark"
)
if _bench_dir.exists() and str(_bench_dir) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_bench_dir))

from .qdp_test_utils import requires_qdp  # noqa: E402


@requires_qdp
def test_benchmark_api_import():
    """Test that the benchmark API exports only Rust-pipeline path (no encode_stream / Python loop)."""
    import api

    assert hasattr(api, "QdpBenchmark")
    assert hasattr(api, "ThroughputResult")
    assert hasattr(api, "LatencyResult")
    # No naive Python for-loop API
    assert not hasattr(api, "encode_stream")
    assert not hasattr(api, "create_pipeline")
    assert not hasattr(api, "StreamPipeline")
    assert not hasattr(api, "PipelineConfig")


@requires_qdp
@pytest.mark.gpu
def test_qdp_benchmark_run_throughput():
    """QdpBenchmark.run_throughput() calls Rust pipeline and returns ThroughputResult (requires GPU)."""
    import api

    result = (
        api.QdpBenchmark(device_id=0)
        .qubits(2)
        .encoding("amplitude")
        .batches(2, size=4)
        .prefetch(2)
        .run_throughput()
    )
    assert isinstance(result, api.ThroughputResult)
    assert result.duration_sec >= 0
    assert result.vectors_per_sec > 0


@requires_qdp
@pytest.mark.gpu
def test_qdp_benchmark_run_latency():
    """QdpBenchmark.run_latency() calls Rust pipeline and returns LatencyResult (requires GPU)."""
    import api

    result = (
        api.QdpBenchmark(device_id=0)
        .qubits(2)
        .encoding("amplitude")
        .batches(2, size=4)
        .prefetch(2)
        .run_latency()
    )
    assert isinstance(result, api.LatencyResult)
    assert result.duration_sec >= 0
    assert result.latency_ms_per_vector > 0


@requires_qdp
def test_qdp_benchmark_validation():
    """QdpBenchmark.run_throughput() raises if qubits/batches not set."""
    import api

    with pytest.raises(ValueError, match="qubits and batches"):
        api.QdpBenchmark(device_id=0).run_throughput()
