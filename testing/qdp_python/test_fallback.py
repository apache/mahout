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
Tests for backend selection when _qdp is unavailable.

The PyTorch reference backend must be explicitly selected via
``.backend("pytorch")``; it is NOT used as an automatic fallback.
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

    def test_auto_detection_skips_pytorch(self):
        """Without _qdp, auto-detection returns NONE, not PYTORCH."""
        from qumat_qdp._backend import Backend, get_backend

        # If _qdp is not installed, get_backend() should be NONE.
        # If _qdp IS installed, it will be RUST_CUDA.  Either way, not PYTORCH.
        b = get_backend()
        assert b is not Backend.PYTORCH

    def test_get_torch(self):
        from qumat_qdp._backend import get_torch

        t = get_torch()
        assert t is not None  # torch is available in test env


# ---------------------------------------------------------------------------
# Loader with explicit PyTorch backend
# ---------------------------------------------------------------------------


class TestLoaderPytorchBackend:
    def test_loader_helpers_cover_iqp_family_edges(self):
        from qumat_qdp.loader import _build_sample, _sample_dim

        assert _sample_dim(3, "basis") == 1
        assert _sample_dim(3, "angle") == 3
        assert _sample_dim(3, "iqp-z") == 3
        assert _sample_dim(3, "iqp") == 6
        assert _sample_dim(3, "amplitude") == 8
        assert _build_sample(4, 0, "iqp") == []
        assert _build_sample(4, 0, "iqp-z") == []

    def test_no_qdp_without_explicit_backend_raises(self, monkeypatch):
        """Without _qdp and without .backend('pytorch'), iteration raises."""
        from qumat_qdp import loader as loader_mod
        from qumat_qdp.loader import QuantumDataLoader

        monkeypatch.setattr(loader_mod, "_get_qdp", lambda: None)
        ld = (
            QuantumDataLoader(device_id=0)
            .qubits(2)
            .encoding("amplitude")
            .batches(1, size=1)
            .source_synthetic()
        )
        with pytest.raises(RuntimeError, match="Rust extension"):
            list(ld)

    def test_synthetic_pytorch_yields_tensors(self):
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .backend("pytorch")
            .qubits(2)
            .encoding("amplitude")
            .batches(3, size=2)
            .source_synthetic()
        )
        batches = list(loader)
        assert len(batches) == 3
        for b in batches:
            assert isinstance(b, torch.Tensor)
            assert b.shape == (2, 4)  # batch_size=2, 2^2=4
            assert b.is_complex()

    def test_synthetic_pytorch_angle(self):
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .backend("pytorch")
            .qubits(3)
            .encoding("angle")
            .batches(2, size=4)
            .source_synthetic()
        )
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0].shape == (4, 8)

    def test_synthetic_pytorch_basis(self):
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .backend("pytorch")
            .qubits(2)
            .encoding("basis")
            .batches(2, size=3)
            .source_synthetic()
        )
        batches = list(loader)
        assert len(batches) == 2
        for b in batches:
            assert b.shape == (3, 4)

    def test_file_npy_pytorch(self, tmp_path):
        import numpy as np
        from qumat_qdp.loader import QuantumDataLoader

        # Create a small .npy file.
        data = np.random.rand(10, 4).astype(np.float64)
        npy_path = str(tmp_path / "test_data.npy")
        np.save(npy_path, data)

        loader = (
            QuantumDataLoader(device_id=0)
            .backend("pytorch")
            .qubits(2)
            .encoding("amplitude")
            .batches(5, size=2)
            .source_file(npy_path)
        )
        batches = list(loader)
        assert len(batches) == 5
        for b in batches:
            assert isinstance(b, torch.Tensor)
            assert b.shape == (2, 4)

    def test_file_parquet_raises(self):
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .backend("pytorch")
            .qubits(2)
            .encoding("amplitude")
            .batches(1, size=1)
            .source_file("data.parquet")
        )
        with pytest.raises(RuntimeError, match="only supports"):
            list(loader)

    def test_synthetic_pytorch_iqp(self):
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .backend("pytorch")
            .qubits(3)
            .encoding("iqp")
            .batches(2, size=4)
            .source_synthetic()
        )
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0].shape == (4, 8)

    def test_synthetic_pytorch_iqp_z(self):
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .backend("pytorch")
            .qubits(3)
            .encoding("iqp-z")
            .batches(2, size=4)
            .source_synthetic()
        )
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0].shape == (4, 8)

    def test_file_pt_pytorch(self, tmp_path):
        from qumat_qdp.loader import QuantumDataLoader

        data = torch.randn(10, 4, dtype=torch.float64)
        pt_path = str(tmp_path / "test_data.pt")
        torch.save(data, pt_path)

        loader = (
            QuantumDataLoader(device_id=0)
            .backend("pytorch")
            .qubits(2)
            .encoding("amplitude")
            .batches(5, size=2)
            .source_file(pt_path)
        )
        batches = list(loader)
        assert len(batches) == 5
        for b in batches:
            assert isinstance(b, torch.Tensor)
            assert b.shape == (2, 4)

    def test_streaming_raises(self):
        from qumat_qdp.loader import QuantumDataLoader

        loader = (
            QuantumDataLoader(device_id=0)
            .backend("pytorch")
            .qubits(2)
            .encoding("amplitude")
            .batches(1, size=1)
            .source_file("data.parquet", streaming=True)
        )
        with pytest.raises(RuntimeError, match="Streaming"):
            list(loader)

    def test_invalid_backend_raises(self):
        from qumat_qdp.loader import QuantumDataLoader

        with pytest.raises(ValueError, match="'rust' or 'pytorch'"):
            QuantumDataLoader(device_id=0).backend("auto")


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

        with pytest.raises(ValueError, match="'rust' or 'pytorch'"):
            QdpBenchmark().backend("invalid")

    def test_auto_backend_raises(self):
        from qumat_qdp.api import QdpBenchmark

        with pytest.raises(ValueError, match="'rust' or 'pytorch'"):
            QdpBenchmark().backend("auto")

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

    @pytest.mark.parametrize("encoding_method", ["iqp", "iqp-z"])
    def test_pytorch_iqp_family(self, encoding_method):
        from qumat_qdp.api import QdpBenchmark

        result = (
            QdpBenchmark()
            .backend("pytorch")
            .qubits(3)
            .encoding(encoding_method)
            .batches(3, size=2)
            .run_throughput()
        )
        assert result.duration_sec > 0
        assert result.vectors_per_sec > 0
