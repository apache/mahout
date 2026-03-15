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
Tests for the fallback mechanism when _qdp is unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


class TestBackendDetection:
    def test_enum_values(self):
        from qumat_qdp._backend import Backend

        assert Backend.RUST_CUDA.value == "rust_cuda"
        assert Backend.PYTORCH.value == "pytorch"
        assert Backend.NONE.value == "none"

    def test_get_backend_returns_valid(self):
        from qumat_qdp._backend import Backend, get_backend

        b = get_backend()
        assert isinstance(b, Backend)

    def test_force_backend(self):
        from qumat_qdp._backend import Backend, force_backend, get_backend

        original = get_backend()
        try:
            force_backend(Backend.PYTORCH)
            assert get_backend() is Backend.PYTORCH
            force_backend(Backend.NONE)
            assert get_backend() is Backend.NONE
        finally:
            force_backend(None)
            assert get_backend() == original

    def test_require_backend_none_raises(self):
        from qumat_qdp._backend import Backend, force_backend, require_backend

        try:
            force_backend(Backend.NONE)
            with pytest.raises(RuntimeError, match="No QDP encoding backend"):
                require_backend()
        finally:
            force_backend(None)

    def test_get_torch(self):
        from qumat_qdp._backend import get_torch

        t = get_torch()
        assert t is not None  # torch is available in test env


# ---------------------------------------------------------------------------
# Loader fallback (simulate _qdp unavailable)
# ---------------------------------------------------------------------------


def _patch_qdp_unavailable(monkeypatch):
    """Make _get_qdp() return None to simulate missing _qdp extension."""
    from qumat_qdp import loader

    # Clear lru_cache and replace with a function returning None.
    monkeypatch.setattr(loader, "_get_qdp", lambda: None)


class TestLoaderFallback:
    def test_synthetic_fallback_yields_tensors(self, monkeypatch):
        _patch_qdp_unavailable(monkeypatch)
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(2)
            .encoding("amplitude")
            .batches(3, size=2)
            .source_synthetic()
        )
        with pytest.warns(UserWarning, match="PyTorch fallback"):
            batches = list(loader)
        assert len(batches) == 3
        for b in batches:
            assert isinstance(b, torch.Tensor)
            assert b.shape == (2, 4)  # batch_size=2, 2^2=4
            assert b.is_complex()

    def test_synthetic_fallback_angle(self, monkeypatch):
        _patch_qdp_unavailable(monkeypatch)
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(3)
            .encoding("angle")
            .batches(2, size=4)
            .source_synthetic()
        )
        with pytest.warns(UserWarning, match="PyTorch fallback"):
            batches = list(loader)
        assert len(batches) == 2
        assert batches[0].shape == (4, 8)

    def test_synthetic_fallback_basis(self, monkeypatch):
        _patch_qdp_unavailable(monkeypatch)
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(2)
            .encoding("basis")
            .batches(2, size=3)
            .source_synthetic()
        )
        with pytest.warns(UserWarning, match="PyTorch fallback"):
            batches = list(loader)
        assert len(batches) == 2
        for b in batches:
            assert b.shape == (3, 4)

    def test_file_npy_fallback(self, monkeypatch, tmp_path):
        _patch_qdp_unavailable(monkeypatch)
        import numpy as np
        from qumat_qdp.loader import QuantumDataLoader

        # Create a small .npy file.
        data = np.random.rand(10, 4).astype(np.float64)
        npy_path = str(tmp_path / "test_data.npy")
        np.save(npy_path, data)

        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(2)
            .encoding("amplitude")
            .batches(5, size=2)
            .source_file(npy_path)
        )
        with pytest.warns(UserWarning, match="PyTorch fallback"):
            batches = list(loader)
        assert len(batches) == 5
        for b in batches:
            assert isinstance(b, torch.Tensor)
            assert b.shape == (2, 4)

    def test_file_parquet_raises(self, monkeypatch):
        _patch_qdp_unavailable(monkeypatch)
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(2)
            .encoding("amplitude")
            .batches(1, size=1)
            .source_file("data.parquet")
        )
        with pytest.warns(UserWarning, match="PyTorch fallback"):
            with pytest.raises(RuntimeError, match="only supports"):
                list(loader)

    def test_synthetic_fallback_iqp(self, monkeypatch):
        _patch_qdp_unavailable(monkeypatch)
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(3)
            .encoding("iqp")
            .batches(2, size=4)
            .source_synthetic()
        )
        with pytest.warns(UserWarning, match="PyTorch fallback"):
            batches = list(loader)
        assert len(batches) == 2
        assert batches[0].shape == (4, 8)

    def test_file_pt_fallback(self, monkeypatch, tmp_path):
        _patch_qdp_unavailable(monkeypatch)
        from qumat_qdp.loader import QuantumDataLoader

        data = torch.randn(10, 4, dtype=torch.float64)
        pt_path = str(tmp_path / "test_data.pt")
        torch.save(data, pt_path)

        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(2)
            .encoding("amplitude")
            .batches(5, size=2)
            .source_file(pt_path)
        )
        with pytest.warns(UserWarning, match="PyTorch fallback"):
            batches = list(loader)
        assert len(batches) == 5
        for b in batches:
            assert isinstance(b, torch.Tensor)
            assert b.shape == (2, 4)

    def test_streaming_raises(self, monkeypatch):
        _patch_qdp_unavailable(monkeypatch)
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(2)
            .encoding("amplitude")
            .batches(1, size=1)
            .source_file("data.parquet", streaming=True)
        )
        with pytest.warns(UserWarning, match="PyTorch fallback"):
            with pytest.raises(RuntimeError, match="Streaming"):
                list(loader)


# ---------------------------------------------------------------------------
# Import-level fallback
# ---------------------------------------------------------------------------


class TestImportFallback:
    def test_backend_exported(self):
        from qumat_qdp import BACKEND, Backend

        assert isinstance(BACKEND, Backend)

    def test_backend_enum_importable(self):
        from qumat_qdp import Backend

        assert hasattr(Backend, "RUST_CUDA")
        assert hasattr(Backend, "PYTORCH")
        assert hasattr(Backend, "NONE")


# ---------------------------------------------------------------------------
# Benchmark API fallback
# ---------------------------------------------------------------------------


class TestBenchmarkFallback:
    def test_backend_builder(self):
        from qumat_qdp.api import QdpBenchmark

        b = QdpBenchmark().backend("pytorch")
        assert b._backend_name == "pytorch"

    def test_invalid_backend_raises(self):
        from qumat_qdp.api import QdpBenchmark

        with pytest.raises(ValueError, match="'auto', 'rust', or 'pytorch'"):
            QdpBenchmark().backend("invalid")

    def test_pytorch_throughput(self):
        from qumat_qdp.api import QdpBenchmark

        result = (
            QdpBenchmark()
            .backend("pytorch")
            .qubits(2)
            .encoding("amplitude")
            .batches(5, size=4)
            .warmup(1)
            .run_throughput()
        )
        assert result.duration_sec > 0
        assert result.vectors_per_sec > 0

    def test_pytorch_latency(self):
        from qumat_qdp.api import QdpBenchmark

        result = (
            QdpBenchmark()
            .backend("pytorch")
            .qubits(2)
            .encoding("amplitude")
            .batches(5, size=4)
            .run_latency()
        )
        assert result.duration_sec > 0
        assert result.latency_ms_per_vector > 0
