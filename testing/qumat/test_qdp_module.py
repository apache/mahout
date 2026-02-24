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

import sys
import importlib
import pytest
from unittest.mock import patch


def _reload_qdp_without_extension():
    """
    Safely reload qumat.qdp while simulating missing _qdp extension.
    Restores sys.modules after execution to avoid test pollution.
    """
    original_qdp = sys.modules.get("qumat.qdp")
    original_ext = sys.modules.get("_qdp")

    try:
        # Remove cached modules
        sys.modules.pop("qumat.qdp", None)
        sys.modules.pop("_qdp", None)

        # Simulate missing compiled extension
        with patch.dict(sys.modules, {"_qdp": None}):
            module = importlib.import_module("qumat.qdp")
            return importlib.reload(module)

    finally:
        # Restore previous state
        if original_qdp is not None:
            sys.modules["qumat.qdp"] = original_qdp
        else:
            sys.modules.pop("qumat.qdp", None)

        if original_ext is not None:
            sys.modules["_qdp"] = original_ext
        else:
            sys.modules.pop("_qdp", None)


def test_qdp_import_fallback_warning():
    with pytest.warns(ImportWarning):
        qdp = _reload_qdp_without_extension()
    assert qdp is not None


def test_qdp_engine_stub_raises_import_error():
    with pytest.warns(ImportWarning):
        qdp = _reload_qdp_without_extension()

    with pytest.raises(ImportError, match="install"):
        qdp.QdpEngine()


def test_quantum_tensor_stub_raises_import_error():
    with pytest.warns(ImportWarning):
        qdp = _reload_qdp_without_extension()

    with pytest.raises(ImportError, match="install"):
        qdp.QuantumTensor()


def test_qdp_all_exports():
    with pytest.warns(ImportWarning):
        qdp = _reload_qdp_without_extension()

    assert "QdpEngine" in qdp.__all__
    assert "QuantumTensor" in qdp.__all__


def test_make_stub_direct_call():
    with pytest.warns(ImportWarning):
        qdp = _reload_qdp_without_extension()

    StubClass = qdp._make_stub("TestStub")

    with pytest.raises(ImportError, match="install"):
        StubClass()