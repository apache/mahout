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

"""Tests for backend detection and builder backend selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from qumat_qdp.loader import QuantumDataLoader

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# GPU availability
# ---------------------------------------------------------------------------


class TestGpuAvailability:
    def test_is_native_cuda_available_returns_bool(self):
        """is_native_cuda_available() returns a plain bool without crashing.

        On a stub build (the extension built without the CUDA toolkit) or a host
        with no device it must be False -- and querying it must NOT abort the
        process, which is the regression this guards (a stub build previously
        aborted when GPU code ran).
        """
        from qumat_qdp import is_native_cuda_available

        assert isinstance(is_native_cuda_available(), bool)

    def test_is_native_cuda_available_matches_extension(self):
        """The helper mirrors the native ``_qdp.cuda_available()`` signal."""
        from qumat_qdp import is_native_cuda_available

        try:
            import _qdp
        except ImportError:
            assert is_native_cuda_available() is False
        else:
            assert is_native_cuda_available() == bool(_qdp.cuda_available())


# ---------------------------------------------------------------------------
# Loader backend selection
# ---------------------------------------------------------------------------


class TestLoaderBackendSelection:
    def test_loader_helpers_cover_iqp_family_edges(self):
        from qumat_qdp import loader as loader_mod

        _build_sample = getattr(loader_mod, "_build_sample")
        _sample_dim = getattr(loader_mod, "_sample_dim")

        assert _sample_dim(3, "basis") == 1
        assert _sample_dim(3, "angle") == 3
        assert _sample_dim(3, "iqp-z") == 3
        assert _sample_dim(3, "iqp") == 6
        assert _sample_dim(3, "amplitude") == 8
        assert _build_sample(4, 0, "iqp") == []
        assert _build_sample(4, 0, "iqp-z") == []

    def test_cuda_backend_without_qdp_raises(self, monkeypatch):
        """The default ``cuda`` backend raises when _qdp is unavailable."""
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

    @staticmethod
    def _auto_loader() -> QuantumDataLoader:
        from qumat_qdp.loader import QuantumDataLoader

        return (
            QuantumDataLoader(device_id=0)
            .backend("auto")
            .qubits(2)
            .encoding("amplitude")
            .batches(1, size=2)
            .source_synthetic()
        )

    def test_backend_names(self):
        from qumat_qdp.loader import QuantumDataLoader

        assert QuantumDataLoader()._backend_name == "cuda"
        assert QuantumDataLoader().backend("cuda")._backend_name == "cuda"
        assert QuantumDataLoader().backend("amd")._backend_name == "amd"
        assert QuantumDataLoader().backend("auto")._backend_name == "auto"

    @pytest.mark.parametrize("name", ["invalid", "rust", "pytorch"])
    def test_invalid_backend_raises(self, name):
        from qumat_qdp.loader import QuantumDataLoader

        with pytest.raises(
            ValueError,
            match=rf"backend must be one of \['amd', 'auto', 'cuda'\], got '{name}'",
        ):
            QuantumDataLoader(device_id=0).backend(name)

    def test_amd_backend_requires_rocm_runtime(self, monkeypatch):
        from qumat_qdp import triton_amd
        from qumat_qdp.loader import QuantumDataLoader

        monkeypatch.setattr(triton_amd, "_is_rocm_runtime", lambda: False)
        ld = (
            QuantumDataLoader(device_id=0)
            .backend("amd")
            .qubits(2)
            .batches(1, size=2)
            .source_synthetic()
        )
        with pytest.raises(RuntimeError, match="Triton AMD backend unavailable"):
            list(ld)

    def test_auto_backend_prefers_rust_cuda(self, monkeypatch):
        """``auto`` picks the Rust/CUDA iterator when CUDA is usable."""
        from types import SimpleNamespace

        from qumat_qdp import loader as loader_mod
        from qumat_qdp import triton_amd

        class FakeEngine:
            def __init__(self, device_id) -> None:
                self.device_id = device_id

            def create_synthetic_loader(self, *, total_batches, **_kwargs: object):
                return ["rust-batch"] * total_batches

        fake_qdp = SimpleNamespace(QdpEngine=FakeEngine, cuda_available=lambda: True)
        monkeypatch.setattr(loader_mod, "_get_qdp", lambda: fake_qdp)
        # AMD being available too must not override the NVIDIA route.
        monkeypatch.setattr(triton_amd, "is_triton_amd_available", lambda: True)
        batches = list(self._auto_loader())
        assert [b.value for b in batches] == ["rust-batch"]
        assert batches[0].backend == "cuda"

    def test_auto_backend_without_gpu_raises(self, monkeypatch):
        """With no GPU backend usable, ``auto`` raises."""
        from qumat_qdp import loader as loader_mod
        from qumat_qdp import triton_amd

        monkeypatch.setattr(loader_mod, "_get_qdp", lambda: None)
        monkeypatch.setattr(triton_amd, "is_triton_amd_available", lambda: False)
        with pytest.raises(RuntimeError, match="No QDP GPU backend is usable"):
            list(self._auto_loader())


# ---------------------------------------------------------------------------
# Benchmark API backend selection
# ---------------------------------------------------------------------------


class TestBenchmarkBackendSelection:
    def test_backend_names(self):
        from qumat_qdp.api import QdpBenchmark

        assert QdpBenchmark()._backend_name == "cuda"
        assert QdpBenchmark()._dtype == "f32"
        assert QdpBenchmark().backend("cuda")._backend_name == "cuda"
        assert QdpBenchmark().backend("amd")._backend_name == "amd"

    @pytest.mark.parametrize("name", ["invalid", "auto", "rust", "pytorch"])
    def test_invalid_backend_raises(self, name):
        from qumat_qdp.api import QdpBenchmark

        with pytest.raises(ValueError, match=rf"'cuda' or 'amd', got '{name}'"):
            QdpBenchmark().backend(name)

    def test_amd_backend_requires_rocm_runtime(self, monkeypatch):
        from qumat_qdp import triton_amd
        from qumat_qdp.api import QdpBenchmark

        monkeypatch.setattr(triton_amd, "_is_rocm_runtime", lambda: False)
        bench = QdpBenchmark().backend("amd").qubits(2).batches(1, size=2)
        with pytest.raises(RuntimeError, match="Triton AMD backend unavailable"):
            bench.run_throughput()
