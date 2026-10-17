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

from __future__ import annotations

import pytest

QdpBenchmark = pytest.importorskip("qumat_qdp").QdpBenchmark


@pytest.mark.parametrize("device_id", [-1, 1.5, "0", None])
def test_device_id_requires_non_negative_integer(device_id: object) -> None:
    with pytest.raises(ValueError, match="device_id must be a non-negative integer"):
        QdpBenchmark(device_id=device_id)


@pytest.mark.parametrize("n", [0, -1, 31, 2.5, "4", None])
def test_qubits_requires_supported_range(n: object) -> None:
    with pytest.raises(ValueError, match="num_qubits must be an integer in"):
        QdpBenchmark().qubits(n)


@pytest.mark.parametrize("n", [1, 30])
def test_qubits_accepts_supported_range(n: int) -> None:
    assert QdpBenchmark().qubits(n)._num_qubits == n


@pytest.mark.parametrize("total", [0, -1, 1.5, "2", None])
def test_batches_requires_positive_total(total: object) -> None:
    with pytest.raises(ValueError, match="total_batches must be a positive integer"):
        QdpBenchmark().batches(total)


@pytest.mark.parametrize("size", [0, -1, 1.5, "2", None])
def test_batches_requires_positive_size(size: object) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        QdpBenchmark().batches(1, size=size)


@pytest.mark.parametrize("n", [-1, 1.5, "2", None])
def test_warmup_requires_non_negative_integer(n: object) -> None:
    with pytest.raises(
        ValueError, match="warmup_batches must be a non-negative integer"
    ):
        QdpBenchmark().warmup(n)


@pytest.mark.parametrize("n", [0, 4])
def test_warmup_accepts_non_negative_integer(n: int) -> None:
    assert QdpBenchmark().warmup(n)._warmup_batches == n


@pytest.mark.parametrize("n", [-1, 1.5, "2", None])
def test_prefetch_requires_non_negative_integer(n: object) -> None:
    with pytest.raises(ValueError, match="prefetch must be a non-negative integer"):
        QdpBenchmark().prefetch(n)


@pytest.mark.parametrize("n", [0, 4])
def test_prefetch_accepts_non_negative_integer(n: int) -> None:
    bench = QdpBenchmark()
    assert bench.prefetch(n) is bench


@pytest.mark.parametrize("method", ["", None, 3, 1.5, b"amplitude"])
def test_encoding_requires_non_empty_string(method: object) -> None:
    with pytest.raises(ValueError, match="encoding_method must be a non-empty string"):
        QdpBenchmark().encoding(method)


@pytest.mark.parametrize("method", ["not-an-encoding", "amplitud", "iqp-y", "angles"])
def test_unknown_encoding_rejected(method: str) -> None:
    with pytest.raises(ValueError, match="Unknown encoding"):
        QdpBenchmark().encoding(method)


@pytest.mark.parametrize(
    "method", ["amplitude", "angle", "basis", "iqp", "iqp-z", "phase"]
)
def test_all_supported_encodings_accepted(method: str) -> None:
    assert QdpBenchmark().encoding(method)._encoding_method == method


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Amplitude", "amplitude"),
        ("ANGLE", "angle"),
        ("IQP-Z", "iqp-z"),
        ("Basis", "basis"),
        ("Iqp", "iqp"),
        ("PHASE", "phase"),
    ],
)
def test_encoding_setter_normalizes_case(given: str, expected: str) -> None:
    assert QdpBenchmark().encoding(given)._encoding_method == expected


def test_valid_configuration_still_runs_on_pytorch_backend() -> None:
    pytest.importorskip("torch")
    result = (
        QdpBenchmark(device_id=0)
        .backend("pytorch")
        .qubits(2)
        .encoding("Amplitude")
        .batches(2, size=4)
        .warmup(1)
        .prefetch(2)
        .run_throughput()
    )
    assert result.duration_sec >= 0
    assert result.vectors_per_sec >= 0
