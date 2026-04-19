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

"""Unified backend routing and DLPack contract for QDP encoders."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any

from qumat_qdp._backend import get_qdp


def _load_qdp_module(required: bool = False) -> Any:
    qdp = get_qdp()
    if qdp is None and required:
        raise RuntimeError(
            "CUDA backend requires compiled _qdp extension. "
            "Build with: uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml"
        )
    return qdp


def _load_triton_backend():
    from qumat_qdp.triton_amd import TritonAmdKernel, is_triton_amd_available

    return TritonAmdKernel, is_triton_amd_available


def _load_unified_qdp_engine():
    qdp = _load_qdp_module(required=False)
    if qdp is None:
        return None
    return getattr(qdp, "QdpEngine", None)


def _create_triton_engine(*, device_id: int, precision: str) -> Any:
    TritonAmdKernel, is_triton_amd_available = _load_triton_backend()
    if not is_triton_amd_available():
        raise RuntimeError("Triton HIP target not detected.")
    engine = TritonAmdKernel(device_id=device_id, precision=precision)
    engine.check_runtime()
    return engine


def _create_amd_engine(*, device_id: int, precision: str) -> tuple[str, Any]:
    qdp_engine = _load_unified_qdp_engine()
    if qdp_engine is not None:
        return "amd", qdp_engine(
            device_id=device_id, precision=precision, backend="amd"
        )

    return "amd", _create_triton_engine(device_id=device_id, precision=precision)


def _rocm_runtime_hint() -> bool:
    # Fast heuristics for ROCm presence without importing torch/triton.
    if not sys.platform.startswith("linux"):
        return False
    if any(
        key in os.environ
        for key in (
            "ROCM_HOME",
            "ROCM_PATH",
            "HIP_VISIBLE_DEVICES",
            "HSA_OVERRIDE_GFX_VERSION",
        )
    ):
        return True
    # Common Ubuntu/ROCm signals.
    if os.path.exists("/opt/rocm"):
        return True
    if os.path.exists("/dev/kfd"):
        return True
    if os.path.exists("/sys/module/amdgpu"):
        return True
    if shutil.which("rocminfo") is not None:
        return True
    return False


@dataclass
class QuantumTensor:
    """
    Unified DLPack producer wrapper.

    Wraps backend-specific outputs (Rust QuantumTensor, torch.Tensor)
    behind a single `__dlpack__`/`__dlpack_device__` contract.
    """

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


class EngineRouter:
    """Select backend and expose a unified `encode -> QuantumTensor` interface."""

    def __init__(
        self, *, backend: str = "auto", device_id: int = 0, precision: str = "float32"
    ) -> None:
        self.device_id = device_id
        self.precision = precision
        self.requested_backend = backend.lower().strip()
        self.backend, self._engine = self._create_backend_engine(
            backend=self.requested_backend,
            device_id=device_id,
            precision=precision,
        )

    @staticmethod
    def _create_backend_engine(
        *, backend: str, device_id: int, precision: str
    ) -> tuple[str, Any]:
        if backend == "auto":
            # Prefer unified AMD routing when ROCm is hinted, then fall back to CUDA.
            rocm_hint = _rocm_runtime_hint()
            if rocm_hint:
                try:
                    return _create_amd_engine(device_id=device_id, precision=precision)
                except Exception as exc:
                    raise RuntimeError(
                        "ROCm environment detected but AMD backend failed to initialize. "
                        "Install the ROCm runtime and either build `_qdp` with AMD support or install Triton HIP support."
                    ) from exc

            qdp = _load_qdp_module(required=False)
            if qdp is not None and getattr(qdp, "QdpEngine", None) is not None:
                return "cuda", qdp.QdpEngine(device_id=device_id, precision=precision)

            # Final chance on Linux: probe the unified AMD route even if heuristics were inconclusive.
            if sys.platform.startswith("linux"):
                try:
                    return _create_amd_engine(device_id=device_id, precision=precision)
                except Exception:
                    pass
            raise RuntimeError(
                'No available backend for auto routing. Install ROCm support for `backend="amd"`, '
                "or build the CUDA extension (`_qdp`)."
            )

        if backend in {"amd", "triton_amd"}:
            try:
                return _create_amd_engine(device_id=device_id, precision=precision)
            except Exception as exc:
                raise RuntimeError(
                    "AMD backend failed to initialize. Install the ROCm runtime and either build `_qdp` with AMD support or install Triton HIP support."
                ) from exc

        if backend == "cuda":
            qdp_engine = _load_unified_qdp_engine()
            if qdp_engine is None:
                raise RuntimeError("_qdp.QdpEngine is unavailable.")
            return "cuda", qdp_engine(
                device_id=device_id, precision=precision, backend="cuda"
            )

        raise ValueError(
            f"Unsupported backend '{backend}'. Use one of: auto, amd, cuda."
        )

    def encode(
        self, data: Any, num_qubits: int, encoding_method: str = "amplitude"
    ) -> QuantumTensor:
        value = self._engine.encode(data, num_qubits, encoding_method)
        if isinstance(value, QuantumTensor):
            return value
        return QuantumTensor(value=value, backend=self.backend)


class QdpEngine(EngineRouter):
    """
    Unified QDP engine facade.

    The public engine API routes to the concrete CUDA or AMD implementation
    underneath, while always returning the same `QuantumTensor` contract.
    """

    def __init__(
        self,
        device_id: int = 0,
        precision: str = "float32",
        backend: str = "auto",
    ) -> None:
        super().__init__(backend=backend, device_id=device_id, precision=precision)


def create_encoder_engine(
    *,
    backend: str = "auto",
    device_id: int = 0,
    precision: str = "float32",
) -> QdpEngine:
    """Create a unified engine router (single input/output contract across backends)."""
    return QdpEngine(device_id=device_id, precision=precision, backend=backend)
