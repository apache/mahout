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

"""Triton AMD backend for QDP encodings on ROCm."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


def _load_optional_module(name: str) -> Any | None:
    try:
        return import_module(name)
    except (
        ImportError
    ):  # pragma: no cover - import failure is surfaced in check_runtime
        return None


torch_mod = _load_optional_module("torch")
triton_mod = _load_optional_module("triton")


def _is_rocm_runtime() -> bool:
    if torch_mod is None:
        return False
    return (
        bool(getattr(torch_mod.version, "hip", None)) and torch_mod.cuda.is_available()
    )


def is_triton_amd_available() -> bool:
    if not _is_rocm_runtime() or triton_mod is None:
        return False
    try:
        target = triton_mod.runtime.driver.active.get_current_target()
        return str(getattr(target, "backend", "")).lower() == "hip"
    except Exception:
        return True


@dataclass
class TritonAmdEngine:
    """AMD backend implementing amplitude/angle/basis encoders."""

    device_id: int = 0
    precision: str = "float32"

    def __post_init__(self) -> None:
        p = self.precision.lower()
        if p in ("float32", "f32", "float"):
            self.precision = "float32"
            return
        if p in ("float64", "f64", "double"):
            self.precision = "float64"
            return
        raise ValueError(
            f"Unsupported precision '{self.precision}'. Use float32 or float64."
        )

    def check_runtime(self) -> None:
        if not _is_rocm_runtime():
            raise RuntimeError(
                "Triton AMD backend unavailable: no PyTorch ROCm device detected."
            )
        if triton_mod is None:
            raise RuntimeError(
                "Triton AMD backend unavailable: install the Triton Python package."
            )

    def _device(self) -> str:
        return f"cuda:{self.device_id}"

    def _require_torch(self) -> Any:
        if torch_mod is None:
            raise RuntimeError(
                "Triton AMD backend unavailable: PyTorch is not installed."
            )
        return torch_mod

    def _real_dtype(self) -> Any:
        torch_mod = self._require_torch()
        return torch_mod.float32 if self.precision == "float32" else torch_mod.float64

    def _complex_dtype(self) -> Any:
        torch_mod = self._require_torch()
        return (
            torch_mod.complex64 if self.precision == "float32" else torch_mod.complex128
        )

    def _to_2d(self, data: Any, *, dtype: Any) -> Any:
        torch_mod = self._require_torch()
        x = torch_mod.as_tensor(data, device=self._device(), dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got {x.ndim}D.")
        return x.contiguous()

    def encode_amplitude(self, data: Any, num_qubits: int) -> Any:
        torch_mod = self._require_torch()
        x = self._to_2d(data, dtype=self._real_dtype())
        _, sample_size = x.shape
        state_len = 1 << num_qubits
        if sample_size != state_len:
            raise ValueError(
                f"Amplitude encoding expects sample size {state_len} (=2^num_qubits), got {sample_size}."
            )

        norms = torch_mod.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)
        amp = x / norms
        return torch_mod.complex(amp, torch_mod.zeros_like(amp)).to(
            self._complex_dtype()
        )

    def encode_angle(self, data: Any, num_qubits: int) -> Any:
        torch_mod = self._require_torch()
        real_dtype = self._real_dtype()
        angles = self._to_2d(data, dtype=real_dtype)
        batch, width = angles.shape
        if width != num_qubits:
            raise ValueError(
                f"Angle encoding expects sample size {num_qubits} (=num_qubits), got {width}."
            )

        state_len = 1 << num_qubits
        idx = torch_mod.arange(state_len, device=angles.device).reshape(1, state_len)
        amp = torch_mod.ones((batch, state_len), device=angles.device, dtype=real_dtype)
        for bit in range(num_qubits):
            col = angles[:, bit].unsqueeze(1)
            factor = torch_mod.where(
                ((idx >> bit) & 1) == 1,
                torch_mod.sin(col),
                torch_mod.cos(col),
            )
            amp = amp * factor

        return torch_mod.complex(amp, torch_mod.zeros_like(amp)).to(
            self._complex_dtype()
        )

    def encode_basis(self, data: Any, num_qubits: int) -> Any:
        torch_mod = self._require_torch()
        idx = torch_mod.as_tensor(data, device=self._device(), dtype=torch_mod.int64)
        if idx.ndim == 2:
            if idx.shape[1] != 1:
                raise ValueError(f"Basis 2D input expects width 1, got {idx.shape[1]}.")
            idx = idx.squeeze(1)
        elif idx.ndim != 1:
            raise ValueError(f"Expected 1D or 2D basis input, got {idx.ndim}D.")

        if idx.numel() == 0:
            raise ValueError("Basis tensor cannot be empty.")

        state_len = 1 << num_qubits
        if torch_mod.any(idx < 0) or torch_mod.any(idx >= state_len):
            raise ValueError(
                f"Basis index out of range. Valid range is [0, {state_len - 1}]."
            )

        batch = int(idx.numel())
        out = torch_mod.zeros(
            (batch, state_len),
            device=idx.device,
            dtype=self._complex_dtype(),
        )
        out.scatter_(
            1,
            idx.reshape(batch, 1),
            torch_mod.ones(
                (batch, 1),
                device=idx.device,
                dtype=self._complex_dtype(),
            ),
        )
        return out

    def encode(
        self,
        data: Any,
        num_qubits: int,
        encoding_method: str = "amplitude",
    ) -> Any:
        self.check_runtime()

        method = encoding_method.lower()
        if method == "amplitude":
            return self.encode_amplitude(data, num_qubits)
        if method == "angle":
            return self.encode_angle(data, num_qubits)
        if method == "basis":
            return self.encode_basis(data, num_qubits)
        raise ValueError(
            f"Unsupported encoding '{encoding_method}'. triton_amd supports amplitude, angle, basis."
        )
