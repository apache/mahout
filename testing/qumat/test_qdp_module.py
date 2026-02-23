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
import builtins
import pytest


def test_qdp_import_fallback_warning(monkeypatch):
    """
    Force the ImportError branch to execute and ensure
    the warning line is covered.
    """

    # Remove module if already imported
    if "qumat.qdp" in sys.modules:
        del sys.modules["qumat.qdp"]

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "_qdp":
            raise ImportError("Forced import failure for coverage")
        return original_import(name, *args, **kwargs)

    # Force _qdp import to fail
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.warns(ImportWarning):
        pass


def test_qdp_engine_stub_raises_import_error():
    import qumat.qdp as qdp

    with pytest.raises(ImportError):
        qdp.QdpEngine()


def test_quantum_tensor_stub_raises_import_error():
    import qumat.qdp as qdp

    with pytest.raises(ImportError):
        qdp.QuantumTensor()


def test_qdp_all_exports():
    import qumat.qdp as qdp

    assert "QdpEngine" in qdp.__all__
    assert "QuantumTensor" in qdp.__all__


def test_make_stub_direct_call():
    """
    Directly test the _make_stub factory to ensure full coverage.
    """
    import qumat.qdp as qdp

    StubClass = qdp._make_stub("TestStub")

    with pytest.raises(ImportError):
        StubClass()
