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

from dataclasses import dataclass
from typing import Any

from qumat_qdp._backend import get_qdp


def _load_qdp_module(required: bool = False) -> Any:
    qdp = get_qdp()
    if qdp is None and required:
        raise RuntimeError(
            "CUDA backend requires the compiled _qdp extension. "
            "Build with: uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml"
        )
    return qdp


def _load_rust_cuda_engine_class(required: bool = False) -> Any:
    qdp = _load_qdp_module(required=required)
    if qdp is None:
        return None
    return getattr(qdp, "QdpEngine", None)


def _load_triton_engine_components():
    from qumat_qdp.triton_amd import TritonAmdEngine, is_triton_amd_available

    return TritonAmdEngine, is_triton_amd_available


@dataclass
class QuantumTensorWrapper:
    """Thin DLPack wrapper for backend-native values."""

    value: Any
    backend: str

    def __dlpack__(self, stream: int | None = None) -> Any:
        if not hasattr(self.value, "__dlpack__"):
            raise RuntimeError(
                f"Backend '{self.backend}' returned object without __dlpack__ support: {type(self.value)!r}"
            )
        if stream is None:
            return self.value.__dlpack__()
        return self.value.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> Any:
        if not hasattr(self.value, "__dlpack_device__"):
            raise RuntimeError(
                f"Backend '{self.backend}' returned object without __dlpack_device__ support: {type(self.value)!r}"
            )
        return self.value.__dlpack_device__()

    def to_torch(self) -> Any:
        import torch

        return torch.from_dlpack(self)


class _CudaEngineAdapter:
    """Adapter for the Rust CUDA engine route."""

    backend = "cuda"

    def __init__(self, *, device_id: int, precision: str) -> None:
        rust_engine_class = _load_rust_cuda_engine_class(required=True)
        if rust_engine_class is None:
            raise RuntimeError("_qdp.QdpEngine is unavailable.")
        self._engine = rust_engine_class(device_id=device_id, precision=precision)

    def encode(
        self, data: Any, num_qubits: int, encoding_method: str = "amplitude"
    ) -> Any:
        return self._engine.encode(data, num_qubits, encoding_method)


class _TritonEngineAdapter:
    """Adapter for the direct Triton AMD engine route."""

    backend = "amd"

    def __init__(self, *, device_id: int, precision: str) -> None:
        triton_engine_class, is_triton_amd_available = _load_triton_engine_components()
        if not is_triton_amd_available():
            raise RuntimeError("Triton HIP target not detected.")
        engine = triton_engine_class(device_id=device_id, precision=precision)
        engine.check_runtime()
        self._engine = engine

    def encode(
        self, data: Any, num_qubits: int, encoding_method: str = "amplitude"
    ) -> Any:
        value = self._engine.encode(data, num_qubits, encoding_method)
        if isinstance(value, QuantumTensorWrapper):
            return value
        return QuantumTensorWrapper(value=value, backend=self.backend)


def _select_engine_adapter(
    *, backend: str, device_id: int, precision: str
) -> tuple[str, Any]:
    requested_backend = backend.lower().strip()

    if requested_backend == "cuda":
        return "cuda", _CudaEngineAdapter(device_id=device_id, precision=precision)

    if requested_backend in {"amd", "triton_amd"}:
        return "amd", _TritonEngineAdapter(
            device_id=device_id, precision=precision
        )

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
        self, data: Any, num_qubits: int, encoding_method: str = "amplitude"
    ) -> Any:
        return self._engine_adapter.encode(data, num_qubits, encoding_method)
