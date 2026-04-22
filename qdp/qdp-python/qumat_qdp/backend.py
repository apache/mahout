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

"""Unified Python facade for explicit QDP backend selection."""

from __future__ import annotations

from typing import Any

from qumat_qdp.backends.cuda import CudaBackendEngine

AmdBackendEngine: Any | None = None


def _load_amd_backend_engine_class() -> type[Any]:
    global AmdBackendEngine
    if AmdBackendEngine is None:
        from qumat_qdp.backends.amd import AmdBackendEngine as cls

        AmdBackendEngine = cls
    return AmdBackendEngine


def _load_triton_engine_type() -> tuple[type[Any], Any]:
    from qumat_qdp.triton_amd import is_triton_amd_available

    return _load_amd_backend_engine_class(), is_triton_amd_available


def _load_rust_cuda_engine_class() -> type[Any]:
    return CudaBackendEngine


def _load_rust_cuda_engine(*, device_id: int, precision: str) -> Any:
    engine_class = _load_rust_cuda_engine_class()
    return engine_class(device_id=device_id, precision=precision)


def _select_engine_adapter(
    backend: str,
    device_id: int,
    precision: str,
) -> tuple[str, Any]:
    requested_backend = backend.lower().strip()

    if requested_backend == "cuda":
        return "cuda", _load_rust_cuda_engine(
            device_id=device_id,
            precision=precision,
        )

    if requested_backend in frozenset({"triton_amd", "amd"}):
        engine_class, _ = _load_triton_engine_type()
        return "amd", engine_class(device_id=device_id, precision=precision)

    raise ValueError(f"Unsupported backend '{backend}'. Use one of: amd, cuda.")


class QdpEngine:
    """Unified Python facade over the CUDA and Triton engine routes."""

    def __init__(
        self,
        device_id: int = 0,
        precision: str = "float32",
        backend: str = "cuda",
    ) -> None:
        self.device_id = device_id
        self.precision = precision
        self.backend, self._engine_adapter = _select_engine_adapter(
            backend=backend,
            device_id=device_id,
            precision=precision,
        )

    def encode(
        self,
        data: Any,
        num_qubits: int,
        encoding_method: str = "amplitude",
    ) -> Any:
        return self._engine_adapter.encode(data, num_qubits, encoding_method)
