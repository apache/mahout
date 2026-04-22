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
import qumat_qdp
from qumat_qdp import QdpEngine
from qumat_qdp import backend as backend_mod


def test_backend_routing_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        QdpEngine(backend="unknown-backend")


def test_create_encoder_engine_instantiates_python_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAdapter:
        def encode(self) -> str:
            return "ok"

    monkeypatch.setattr(
        backend_mod,
        "_select_engine_adapter",
        lambda *args, **kwargs: ("fake", FakeAdapter()),
    )

    engine = QdpEngine(backend="cuda")
    assert isinstance(engine, QdpEngine)
    assert engine.backend == "fake"


def test_cuda_route_uses_rust_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeRustEngine:
        def __init__(self, device_id: int = 0, precision: str = "float32") -> None:
            seen["init"] = {"device_id": device_id, "precision": precision}

        def encode(self, data: Any, num_qubits: int, encoding_method: str) -> str:
            seen["encode"] = {
                "data": data,
                "num_qubits": num_qubits,
                "encoding_method": encoding_method,
            }
            return "rust-quantum-tensor"

    monkeypatch.setattr(
        backend_mod, "_load_rust_cuda_engine_class", lambda: FakeRustEngine
    )

    engine = QdpEngine(device_id=3, precision="float64", backend="cuda")
    value = engine.encode([1.0, 0.0], 1, "amplitude")

    assert engine.backend == "cuda"
    assert value == "rust-quantum-tensor"
    assert seen["init"] == {"device_id": 3, "precision": "float64"}
    assert seen["encode"]["encoding_method"] == "amplitude"


def test_amd_route_returns_triton_output_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeValue:
        def __dlpack__(self, stream: Any | None = None) -> Any:
            return ("capsule", stream)

        def __dlpack_device__(self) -> Any:
            return (2, 0)

    class FakeTritonEngine:
        def __init__(self, device_id: int = 0, precision: str = "float32") -> None:
            self.device_id = device_id
            self.precision = precision

        def check_runtime(self) -> None:
            return None

        def encode(self, data: Any, num_qubits: int, encoding_method: str) -> Any:
            return FakeValue()

    monkeypatch.setattr(
        backend_mod,
        "_load_triton_engine_type",
        lambda: (FakeTritonEngine, lambda: True),
    )

    engine = QdpEngine(device_id=1, precision="float32", backend="amd")
    qt = engine.encode([1.0, 0.0], 1, "amplitude")

    assert engine.backend == "amd"
    assert isinstance(qt, FakeValue)
    assert qt.__dlpack_device__() == (2, 0)


def test_missing_cuda_backend_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingCudaBackendEngine:
        def __init__(self, device_id: int, precision: str) -> None:
            raise RuntimeError("_qdp.QdpEngine is unavailable.")

    monkeypatch.setattr(backend_mod, "CudaBackendEngine", MissingCudaBackendEngine)

    with pytest.raises(RuntimeError, match="_qdp.QdpEngine is unavailable"):
        QdpEngine(backend="cuda")


def test_quantum_tensor_wrapper_requires_dlpack() -> None:
    with pytest.raises(RuntimeError):
        qumat_qdp.QdpTensor(value=object(), backend="fake").__dlpack__()


def test_package_exports_quantum_tensor_contract() -> None:
    assert hasattr(qumat_qdp, "QuantumTensor")
    assert hasattr(qumat_qdp, "QdpTensor")
