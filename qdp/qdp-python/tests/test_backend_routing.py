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

from typing import Any

import pytest
from qumat_qdp import QdpEngine, create_encoder_engine
from qumat_qdp.backend import EngineRouter, QuantumTensor


def test_backend_routing_rejects_unknown_backend():
    with pytest.raises(ValueError):
        create_encoder_engine(backend="unknown-backend")


def test_create_encoder_engine_returns_unified_qdp_engine(monkeypatch):
    monkeypatch.setattr(
        EngineRouter,
        "_create_backend_engine",
        staticmethod(lambda **kwargs: ("fake", object())),
    )
    engine = create_encoder_engine(backend="auto")
    assert isinstance(engine, QdpEngine)


def test_qdp_engine_wraps_backend_output_into_unified_quantum_tensor(monkeypatch):
    class FakeValue:
        def __dlpack__(self, stream: Any | None = None) -> Any:
            return ("capsule", stream)

        def __dlpack_device__(self) -> Any:
            return (2, 0)

    class FakeBackendEngine:
        def encode(self, data, num_qubits, encoding_method):
            return FakeValue()

    monkeypatch.setattr(
        EngineRouter,
        "_create_backend_engine",
        staticmethod(lambda **kwargs: ("fake", FakeBackendEngine())),
    )

    engine = QdpEngine(device_id=0, precision="float32", backend="auto")
    qt = engine.encode([1.0, 0.0, 0.0, 0.0], 2, "amplitude")

    assert isinstance(qt, QuantumTensor)
    assert qt.backend == "fake"
    assert qt.__dlpack_device__() == (2, 0)


def test_auto_router_without_available_backends_fails_cleanly():
    # On environments without _qdp / Triton accelerators, auto should fail
    # with an actionable runtime error (instead of importing CUDA eagerly).
    with pytest.raises(RuntimeError, match="No available backend"):
        create_encoder_engine(backend="auto")


def test_unified_quantum_tensor_requires_dlpack():
    with pytest.raises(RuntimeError):
        QuantumTensor(value=object(), backend="fake").__dlpack__()
