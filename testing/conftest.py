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
- Custom pytest markers (gpu, slow)
- Auto-skip logic for QDP tests when the native extension is not built
- Shared fixtures for QDP availability checking

QDP tests are automatically skipped if the _qdp extension is not available,
allowing contributors without CUDA to run the qumat test suite.
"""

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
    This mirrors the ``torch.cuda.is_available()`` guard used by the inline
    skips throughout the QDP test modules.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


_GPU_AVAILABLE = _gpu_available()


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU and _qdp extension"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running")


def pytest_collection_modifyitems(config, items):
    """Auto-skip QDP tests when the extension is missing, and GPU tests when
    no CUDA device is available.

    ``_qdp`` can now build/import without a GPU (stub CUDA symbols), so
    extension availability alone no longer implies a usable device.  GPU tests
    are therefore skipped whenever ``torch.cuda.is_available()`` is False, even
    when the extension imports -- otherwise they execute against the stub
    runtime and abort the worker process.
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
    }

    for item in items:
        is_gpu = "gpu" in item.keywords

        fspath_str = str(item.fspath)
        needs_qdp = (
            "testing/qdp" in fspath_str or "testing\\qdp" in fspath_str
        ) and not any(name in fspath_str for name in _NO_QDP_OK)

        if not _QDP_AVAILABLE:
            # No extension at all: skip GPU tests and everything needing _qdp.
            if is_gpu or needs_qdp:
                item.add_marker(skip_no_qdp)
        elif is_gpu and not _GPU_AVAILABLE:
            # Extension built (possibly with CUDA stubs) but no usable device.
            item.add_marker(skip_no_gpu)


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

    Automatically skips if QDP is not available.

    Usage:
        def test_encoding(qdp_engine):
            qtensor = qdp_engine.encode([1.0, 2.0], num_qubits=1, encoding_method="amplitude")
            ...
    """
    from _qdp import QdpEngine

    return QdpEngine(0)
