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


"""Unit tests for the GPU skip helpers in testing/conftest.py (#1331).

These run on any machine: ``torch.cuda`` is monkeypatched to impersonate a
GPU host, so no GPU (and no ``_qdp`` extension) is needed.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from qumat_qdp.loader import _select_torch_device

from testing import conftest

# Arch list of a typical CUDA 12.8 PyTorch wheel (no Pascal/sm_6x kernels).
_WHEEL_ARCHS = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def _fake_gpu(monkeypatch, capability=(8, 6), arch_list=None, count=1):
    """Make torch.cuda report ``count`` GPUs of the given capability."""
    archs = list(_WHEEL_ARCHS if arch_list is None else arch_list)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: archs)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "FakeGPU")


def test_no_cuda_is_incompatible(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert conftest._cuda_compatible_with_torch() is False
    reason = conftest._torch_cuda_skip_reason()
    assert reason is not None
    assert "CUDA not available" in reason


def test_supported_arch_is_compatible(monkeypatch):
    _fake_gpu(monkeypatch, capability=(8, 6))
    assert conftest._cuda_compatible_with_torch() is True
    assert conftest._torch_cuda_skip_reason() is None


def test_unsupported_arch_skips_with_clear_reason(monkeypatch):
    # Pascal (sm_61) against a wheel that only ships sm_70+ kernels: the
    # situation that produces cudaErrorNoKernelImageForDevice.
    _fake_gpu(monkeypatch, capability=(6, 1))
    assert conftest._cuda_compatible_with_torch() is False
    reason = conftest._torch_cuda_skip_reason()
    assert reason is not None
    assert reason.startswith("GPU arch not supported by this PyTorch build")
    assert "sm_61" in reason


def test_unsupported_arch_does_not_warn(monkeypatch):
    # _select_torch_device warns before falling back; the helper is called at
    # collection time and must stay quiet (the skip reason says the same).
    _fake_gpu(monkeypatch, capability=(6, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert conftest._cuda_compatible_with_torch() is False


def test_empty_arch_list_is_compatible(monkeypatch):
    # Same rule as production: an empty arch list means "unknown", not "none".
    _fake_gpu(monkeypatch, capability=(6, 1), arch_list=[])
    assert conftest._cuda_compatible_with_torch() is True


def test_out_of_range_device_is_incompatible(monkeypatch):
    _fake_gpu(monkeypatch, count=1)
    assert conftest._cuda_compatible_with_torch(device_id=1) is False
    reason = conftest._torch_cuda_skip_reason(device_id=1)
    assert reason is not None
    assert "probe failed" in reason


@pytest.mark.parametrize(
    ("capability", "arch_list"),
    [
        ((8, 6), _WHEEL_ARCHS),
        ((6, 1), _WHEEL_ARCHS),
        ((7, 0), ["sm_70"]),
        ((8, 9), ["sm_80", "sm_86"]),
        ((6, 1), []),
    ],
)
def test_matches_production_device_selection(monkeypatch, capability, arch_list):
    """The helper skips exactly when the loader would fall back to CPU."""
    _fake_gpu(monkeypatch, capability=capability, arch_list=arch_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        production_uses_gpu = _select_torch_device(torch, 0) != "cpu"
    assert conftest._cuda_compatible_with_torch() is production_uses_gpu


class _FakeItem:
    """Just enough of a pytest Item for pytest_collection_modifyitems."""

    def __init__(self, *markers: str, path: str = "testing/qdp/test_x.py") -> None:
        self.keywords = {m: True for m in markers}
        self.fspath = path
        self._markers = set(markers)
        self.added: list = []

    def get_closest_marker(self, name):
        return SimpleNamespace(name=name) if name in self._markers else None

    def add_marker(self, marker):
        self.added.append(marker)

    def skip_reason(self) -> str:
        """Return the added skip reason, or "" if the item was not skipped."""
        for m in self.added:
            if m.name == "skip":
                return m.kwargs["reason"]
        return ""


def _collect(monkeypatch, items, *, qdp=True, gpu=True, torch_reason=None):
    monkeypatch.setattr(conftest, "_QDP_AVAILABLE", qdp)
    monkeypatch.setattr(conftest, "_GPU_AVAILABLE", gpu)
    monkeypatch.setattr(
        conftest, "_torch_cuda_skip_reason", lambda device_id=0: torch_reason
    )
    conftest.pytest_collection_modifyitems(None, items)


def test_collection_skips_gpu_torch_on_unsupported_arch(monkeypatch):
    plain, gpu_only, both = _FakeItem(), _FakeItem("gpu"), _FakeItem("gpu", "gpu_torch")
    reason = "GPU arch not supported by this PyTorch build: sm_61"
    _collect(monkeypatch, [plain, gpu_only, both], torch_reason=reason)
    assert plain.skip_reason() == ""
    assert gpu_only.skip_reason() == ""
    assert both.skip_reason() == reason


def test_collection_prefers_gpu_reason_when_no_device(monkeypatch):
    both = _FakeItem("gpu", "gpu_torch")
    _collect(monkeypatch, [both], gpu=False, torch_reason="CUDA not available")
    assert both.skip_reason().startswith("GPU required")


def test_collection_runs_gpu_torch_when_supported(monkeypatch):
    both = _FakeItem("gpu", "gpu_torch")
    torch_only = _FakeItem("gpu_torch", path="testing/qdp_python/test_torch_ref.py")
    _collect(monkeypatch, [both, torch_only], qdp=False, torch_reason=None)
    # Without _qdp, the gpu test is skipped; the torch-only test still runs.
    assert both.skip_reason().startswith("QDP extension not available")
    assert torch_only.skip_reason() == ""
