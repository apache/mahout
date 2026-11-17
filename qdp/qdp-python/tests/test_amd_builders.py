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

"""ROCm tests for the ``"amd"`` backend of ``QuantumDataLoader`` and ``QdpBenchmark``."""

import numpy as np
import pytest
import torch
from qumat_qdp import (
    QdpBenchmark,
    QdpTensor,
    QuantumDataLoader,
    is_triton_amd_available,
)

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.skipif(
        not is_triton_amd_available(), reason="Triton AMD backend unavailable"
    ),
]


def _loader(encoding: str, num_qubits: int) -> QuantumDataLoader:
    return (
        QuantumDataLoader(device_id=0)
        .backend("amd")
        .qubits(num_qubits)
        .encoding(encoding)
        .batches(2, size=4)
    )


@pytest.mark.parametrize(
    ("encoding", "num_qubits"),
    [("amplitude", 3), ("angle", 3), ("basis", 2), ("iqp", 3), ("iqp-z", 3)],
)
def test_loader_synthetic_yields_device_tensors(encoding: str, num_qubits: int) -> None:
    batches = list(_loader(encoding, num_qubits).source_synthetic())
    assert len(batches) == 2
    for batch in batches:
        assert isinstance(batch, QdpTensor)
        assert batch.backend == "amd"
        out = torch.from_dlpack(batch)
        assert out.device.type == "cuda"
        assert out.shape == (4, 1 << num_qubits)
        assert out.dtype == torch.complex64  # synthetic data is generated in float32


def test_loader_file_precision(tmp_path) -> None:
    npy_path = str(tmp_path / "data.npy")
    np.save(npy_path, np.random.rand(8, 8))

    lossless = list(_loader("amplitude", 3).source_file(npy_path))
    narrowed = list(_loader("amplitude", 3).source_file(npy_path).dtype("float32"))
    assert len(lossless) == len(narrowed) == 2
    assert torch.from_dlpack(lossless[0]).dtype == torch.complex128
    assert torch.from_dlpack(narrowed[0]).dtype == torch.complex64


def test_loader_file_parquet_raises(tmp_path) -> None:
    parquet_path = tmp_path / "data.parquet"
    parquet_path.touch()
    with pytest.raises(RuntimeError, match="only supports"):
        list(_loader("amplitude", 3).source_file(str(parquet_path)))


def test_auto_picks_amd_without_native_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from qumat_qdp import loader as loader_mod

    monkeypatch.setattr(loader_mod, "_get_qdp", lambda: None)
    loader = (
        QuantumDataLoader(device_id=0)
        .backend("auto")
        .qubits(2)
        .batches(1, size=2)
        .source_synthetic()
    )
    batch = next(iter(loader))
    assert batch.backend == "amd"
    assert torch.from_dlpack(batch).device.type == "cuda"


def test_as_torch_dataset_yields_tensors() -> None:
    dataset = _loader("amplitude", 3).source_synthetic().as_torch_dataset()
    batches = list(dataset)
    assert len(batches) == 2
    assert all(isinstance(b, torch.Tensor) and b.is_cuda for b in batches)


@pytest.mark.parametrize("encoding", ["amplitude", "angle", "basis", "iqp", "iqp-z"])
def test_benchmark_throughput_and_latency(encoding: str) -> None:
    bench = (
        QdpBenchmark(device_id=0)
        .backend("amd")
        .qubits(6)
        .encoding(encoding)
        .batches(3, size=8)
        .warmup(1)
    )
    throughput = bench.run_throughput()
    assert throughput.duration_sec > 0
    assert throughput.vectors_per_sec > 0
    latency = bench.run_latency()
    assert latency.latency_ms_per_vector > 0


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_benchmark_dtype_accepted(dtype: str) -> None:
    result = (
        QdpBenchmark(device_id=0)
        .backend("amd")
        .dtype(dtype)
        .qubits(4)
        .encoding("amplitude")
        .batches(2, size=4)
        .run_throughput()
    )
    assert result.vectors_per_sec > 0
