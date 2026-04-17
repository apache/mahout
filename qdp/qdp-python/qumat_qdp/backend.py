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

"""Backend routing and output contract for QDP encoders."""

from __future__ import annotations

import os
from dataclasses import dataclass
import shutil
import sys
from typing import Any


def _load_qdp_module(required: bool = False):
    try:
        import _qdp

        return _qdp
    except Exception as exc:
        if required:
            raise RuntimeError(
                "CUDA backend requires compiled _qdp extension. "
                "Build with: uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml"
            ) from exc
        return None


def _load_triton_backend():
    from qumat_qdp.triton_amd import TritonAmdEngine, is_triton_amd_available

    return TritonAmdEngine, is_triton_amd_available


def _rocm_runtime_hint() -> bool:
    # Fast heuristics for ROCm presence without importing torch/triton.
    if not sys.platform.startswith("linux"):
        return False
    if any(
        key in os.environ
        for key in ("ROCM_HOME", "ROCM_PATH", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION")
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
            # Prefer vendor-neutral/non-CUDA paths first.
            rocm_hint = _rocm_runtime_hint()
            if rocm_hint:
                try:
                    TritonAmdEngine, is_triton_amd_available = _load_triton_backend()
                    if is_triton_amd_available():
                        return "triton_amd", TritonAmdEngine(device_id=device_id, precision=precision)
                except Exception:
                    pass

            if rocm_hint:
                raise RuntimeError(
                    "ROCm environment detected but triton_amd backend is unavailable. "
                    "Install Triton HIP support."
                )

            qdp = _load_qdp_module(required=False)
            if qdp is not None and getattr(qdp, "QdpEngine", None) is not None:
                return "cuda", qdp.QdpEngine(device_id=device_id, precision=precision)

            # Final chance on Linux: try Triton probe even if hint was inconclusive.
            if sys.platform.startswith("linux"):
                try:
                    TritonAmdEngine, is_triton_amd_available = _load_triton_backend()
                    if is_triton_amd_available():
                        return "triton_amd", TritonAmdEngine(device_id=device_id, precision=precision)
                except Exception:
                    pass
            raise RuntimeError(
                "No available backend for auto routing. Install Triton+ROCm (`triton_amd`), "
                "or build CUDA extension (`_qdp`)."
            )

        if backend == "triton_amd":
            TritonAmdEngine, _ = _load_triton_backend()
            engine = TritonAmdEngine(device_id=device_id, precision=precision)
            engine.check_runtime()
            return "triton_amd", engine

        if backend == "cuda":
            qdp = _load_qdp_module(required=True)
            if getattr(qdp, "QdpEngine", None) is None:
                raise RuntimeError("_qdp.QdpEngine is unavailable.")
            return "cuda", qdp.QdpEngine(device_id=device_id, precision=precision)

        raise ValueError(
            f"Unsupported backend '{backend}'. Use one of: auto, triton_amd, cuda."
        )

    def encode(self, data: Any, num_qubits: int, encoding_method: str = "amplitude") -> QuantumTensor:
        value = self._engine.encode(data, num_qubits, encoding_method)
        if isinstance(value, QuantumTensor):
            return value
        return QuantumTensor(value=value, backend=self.backend)


def create_encoder_engine(
    *,
    backend: str = "auto",
    device_id: int = 0,
    precision: str = "float32",
) -> EngineRouter:
    """Create a unified engine router (single input/output contract across backends)."""
    return EngineRouter(backend=backend, device_id=device_id, precision=precision)
