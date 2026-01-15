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
Root pytest configuration for Apache Mahout.

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
_QDP_IMPORT_ERROR = None

try:
    import _qdp

    _QDP_AVAILABLE = True
except ImportError as e:
    _QDP_IMPORT_ERROR = str(e)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU and _qdp extension"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU/QDP tests if the _qdp extension is not available."""
    if _QDP_AVAILABLE:
        return

    skip_marker = pytest.mark.skip(
        reason=f"QDP extension not available: {_QDP_IMPORT_ERROR}. "
        "Build with: cd qdp/qdp-python && maturin develop"
    )

    for item in items:
        # Skip tests explicitly marked with @pytest.mark.gpu
        if "gpu" in item.keywords:
            item.add_marker(skip_marker)

        # Skip all tests in testing/qdp/ directory
        fspath_str = str(item.fspath)
        if "testing/qdp" in fspath_str or "testing\\qdp" in fspath_str:
            item.add_marker(skip_marker)


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
