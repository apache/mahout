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
Shared pytest configuration and fixtures for all tests.

This module provides:
- Custom pytest markers (gpu, gpu_torch, slow)
- Auto-skip logic for QDP tests when the native extension is not built
- Auto-skip logic for ``gpu`` tests when no CUDA device is usable, and for
  ``gpu_torch`` tests when PyTorch cannot run kernels on the GPU
- Shared fixtures for QDP availability checking

QDP tests are automatically skipped if the _qdp extension is not available,
allowing contributors without CUDA to run the qumat test suite.
"""

import warnings

import pytest

# Check if QDP extension is available at module load time
_QDP_AVAILABLE = False
_QDP_IMPORT_ERROR: str | None = "No module named '_qdp'"
try:
    import _qdp

    _QDP_AVAILABLE = True
    _QDP_IMPORT_ERROR = None
except ImportError as e:
    _QDP_IMPORT_ERROR = str(e)


def _gpu_available() -> bool:
    """Return True if a CUDA device is actually usable at runtime.

    The ``_qdp`` extension can now be built and imported without the CUDA
    toolkit -- it links stub CUDA Runtime symbols (see qdp-core ``build.rs`` /
    ``cuda_ffi.rs``).  So importing ``_qdp`` no longer implies a working GPU,
    and ``@pytest.mark.gpu`` tests must additionally check for a real device.

    Prefers the native engine's own signal (``qumat_qdp.is_cuda_available()``),
    which is correct even for a stub build: a host with a GPU + PyTorch but no
    CUDA toolkit builds ``_qdp`` against stubs, where ``torch.cuda.is_available()``
    would wrongly report True. Falls back to torch when the helper is absent.
    """
    try:
        from qumat_qdp import is_cuda_available

        return bool(is_cuda_available())
    except Exception:
        pass
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


_GPU_AVAILABLE = _gpu_available()


def _torch_cuda_skip_reason(device_id: int = 0) -> str | None:
    """Return why PyTorch cannot run CUDA kernels on ``device_id``, or None.

    ``torch.cuda.is_available()`` is True whenever a driver and a GPU are
    present, but the PyTorch wheel only ships kernels for the compute
    capabilities in ``torch.cuda.get_arch_list()``.  On other GPUs (e.g.
    Pascal ``sm_61`` with a recent wheel) the first kernel launch fails with
    ``cudaErrorNoKernelImageForDevice``.

    This delegates to ``qumat_qdp.loader._select_torch_device`` -- the check
    production code uses (PR #1323) -- so tests skip exactly when the loader
    would fall back to CPU.  It needs only PyTorch and the pure-Python
    ``qumat_qdp`` package, not the ``_qdp`` extension.
    """
    try:
        import torch
    except Exception as e:
        return f"PyTorch not available: {e}"
    try:
        if not torch.cuda.is_available():
            return "CUDA not available to PyTorch"
        from qumat_qdp.loader import _select_torch_device

        with warnings.catch_warnings():
            # _select_torch_device warns before falling back to CPU; the skip
            # reason below carries the same information.
            warnings.simplefilter("ignore")
            device = _select_torch_device(torch, device_id)
        if device.startswith("cuda"):
            return None
        major, minor = torch.cuda.get_device_capability(device_id)
        return (
            "GPU arch not supported by this PyTorch build: "
            f"sm_{major}{minor} not in {sorted(torch.cuda.get_arch_list())}"
        )
    except Exception as e:
        # ValueError for an out-of-range device_id, ImportError if qumat_qdp
        # is not installed, RuntimeError from a broken CUDA install, ...
        return f"PyTorch CUDA probe failed: {e}"


def _cuda_compatible_with_torch(device_id: int = 0) -> bool:
    """Return True iff PyTorch can launch CUDA kernels on ``device_id``.

    Mirrors ``qumat_qdp.loader._select_torch_device`` (device capability
    intersected with ``torch.cuda.get_arch_list()``); see
    ``_torch_cuda_skip_reason`` for details.
    """
    return _torch_cuda_skip_reason(device_id) is None


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU and _qdp extension"
    )
    config.addinivalue_line(
        "markers",
        "gpu_torch: marks tests that run PyTorch CUDA ops (device='cuda'); "
        "skipped unless PyTorch supports the GPU's compute capability",
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running")


def pytest_collection_modifyitems(config, items):
    """Auto-skip QDP tests when the extension is missing, GPU tests when no
    CUDA device is available, and PyTorch-on-CUDA tests when the GPU's arch
    is not supported by the installed PyTorch build.

    ``_qdp`` can now build/import without a GPU (stub CUDA symbols), so
    extension availability alone no longer implies a usable device.  ``gpu``
    tests are therefore skipped whenever the native engine reports no device
    (see ``_gpu_available``), even when the extension imports -- otherwise
    they execute against the stub runtime and abort the worker process.

    ``gpu_torch`` is independent of ``gpu``: it guards tests that create
    tensors on ``device="cuda"`` or run PyTorch ops on GPU tensors.  A test
    that needs both the native engine and PyTorch CUDA kernels carries both
    markers; the ``gpu`` skip reason wins when both apply.
    """
    skip_no_qdp = pytest.mark.skip(
        reason=f"QDP extension not available: {_QDP_IMPORT_ERROR}. "
        "Build with: cd qdp/qdp-python && maturin develop"
    )
    skip_no_gpu = pytest.mark.skip(
        reason="GPU required: no CUDA device available. The _qdp extension is "
        "importable but linked against CUDA stubs, or no GPU is present."
    )

    # Tests that work without _qdp (PyTorch reference backend tests).
    _NO_QDP_OK = {
        "test_torch_ref.py",
        "test_fallback.py",
        "test_benchmark_utils.py",
        "test_benchmark_cli_validation.py",
        "test_gpu_guards.py",
    }

    # Computed lazily: importing torch is slow, and most runs (e.g. the qumat
    # suite) contain no gpu_torch tests at all.
    torch_cuda_reason: str | None = None
    torch_cuda_probed = False

    for item in items:
        is_gpu = "gpu" in item.keywords

        fspath_str = str(item.fspath)
        needs_qdp = (
            "testing/qdp" in fspath_str or "testing\\qdp" in fspath_str
        ) and not any(name in fspath_str for name in _NO_QDP_OK)

        if not _QDP_AVAILABLE and (is_gpu or needs_qdp):
            # No extension at all: skip GPU tests and everything needing _qdp.
            item.add_marker(skip_no_qdp)
        elif is_gpu and not _GPU_AVAILABLE:
            # Extension built (possibly with CUDA stubs) but no usable device.
            item.add_marker(skip_no_gpu)
        elif item.get_closest_marker("gpu_torch") is not None:
            if not torch_cuda_probed:
                torch_cuda_reason = _torch_cuda_skip_reason()
                torch_cuda_probed = True
            if torch_cuda_reason is not None:
                item.add_marker(pytest.mark.skip(reason=torch_cuda_reason))


@pytest.fixture
def qdp_available():
    """
    Fixture that skips the test if QDP extension is not available.

    Usage:
        def test_something_with_qdp(qdp_available):
            from _qdp import QdpEngine
            engine = QdpEngine(0)
            ...
    """
    if not _QDP_AVAILABLE:
        pytest.skip(f"QDP extension not available: {_QDP_IMPORT_ERROR}")
    return True


@pytest.fixture
def qdp_engine(qdp_available):
    """
    Fixture that provides a QDP engine instance.

    Automatically skips if QDP is not available, or if the extension is
    importable but no CUDA device is usable (e.g. a stub build), since
    constructing ``QdpEngine`` would then fail or abort the process.

    Usage:
        def test_encoding(qdp_engine):
            qtensor = qdp_engine.encode([1.0, 2.0], num_qubits=1, encoding_method="amplitude")
            ...
    """
    if not _GPU_AVAILABLE:
        pytest.skip("GPU required: no CUDA device available for QdpEngine")
    from _qdp import QdpEngine

    return QdpEngine(0)
