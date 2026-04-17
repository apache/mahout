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
from typing import Any

import torch
import triton
import triton.language as tl


def _is_rocm_runtime() -> bool:
    return bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()


def is_triton_amd_available() -> bool:
    """Return True when Triton is available and active target is HIP/ROCm."""
    if not _is_rocm_runtime():
        return False
    try:
        target = triton.runtime.driver.active.get_current_target()
        return str(getattr(target, "backend", "")).lower() == "hip"
    except Exception:
        # Conservative fallback: ROCm torch build + triton import available.
        return True


@triton.jit
def _amplitude_pack_kernel(
    input_ptr,
    inv_norms_ptr,
    out_ptr,
    sample_size,
    state_len,
    total_elems,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems

    sample_idx = offs // state_len
    elem_idx = offs % state_len
    in_idx = sample_idx * sample_size + elem_idx

    vals = tl.load(input_ptr + in_idx, mask=mask & (elem_idx < sample_size), other=0.0)
    inv = tl.load(inv_norms_ptr + sample_idx, mask=mask, other=0.0)
    real = vals * inv

    out_base = offs * 2
    tl.store(out_ptr + out_base, real, mask=mask)
    tl.store(out_ptr + out_base + 1, 0.0, mask=mask)


@triton.jit
def _angle_kernel(
    angles_ptr,
    out_ptr,
    state_len,
    total_elems,
    NQ: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems

    sample_idx = offs // state_len
    elem_idx = offs % state_len

    amp_dtype = tl.float64 if FP64 else tl.float32
    if NQ == 0:
        amp = tl.full([BLOCK], 1.0, dtype=amp_dtype)
    else:
        bit_offsets = tl.arange(0, NQ)
        angles_base = tl.expand_dims(sample_idx * NQ, 1)
        bit_offsets_2d = tl.expand_dims(bit_offsets, 0)
        angle_ptrs = angles_ptr + angles_base + bit_offsets_2d
        angle_mask = tl.expand_dims(mask, 1)
        angles = tl.load(angle_ptrs, mask=angle_mask, other=0.0)
        elem_idx_2d = tl.expand_dims(elem_idx, 1)
        bit_sets = ((elem_idx_2d >> bit_offsets_2d) & 1) != 0
        factors = tl.where(bit_sets, tl.sin(angles), tl.cos(angles)).to(amp_dtype)
        amp = tl.reduce(factors, axis=1, combine_fn=_product_reduce)

    out_base = offs * 2
    tl.store(out_ptr + out_base, amp, mask=mask)
    tl.store(out_ptr + out_base + 1, 0.0, mask=mask)


@triton.jit
def _product_reduce(a, b):
    return a * b


@triton.jit
def _basis_kernel(
    basis_idx_ptr,
    out_ptr,
    state_len,
    total_elems,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems

    sample_idx = offs // state_len
    elem_idx = offs % state_len
    idx = tl.load(basis_idx_ptr + sample_idx, mask=mask, other=0)
    is_one = elem_idx == idx

    out_base = offs * 2
    one = tl.full([BLOCK], 1.0, dtype=tl.float64 if FP64 else tl.float32)
    zero = tl.full([BLOCK], 0.0, dtype=tl.float64 if FP64 else tl.float32)
    tl.store(out_ptr + out_base, tl.where(is_one, one, zero), mask=mask)
    tl.store(out_ptr + out_base + 1, zero, mask=mask)


@dataclass
class TritonAmdEngine:
    """Triton/ROCm backend implementing amplitude, angle, and basis encodings."""

    device_id: int = 0
    precision: str = "float32"

    def __post_init__(self) -> None:
        p = self.precision.lower()
        if p not in ("float32", "f32", "float", "float64", "f64", "double"):
            raise ValueError(
                f"Unsupported precision '{self.precision}'. Use float32 or float64."
            )
        self.precision = "float32" if p in ("float32", "f32", "float") else "float64"

    @property
    def backend(self) -> str:
        return "triton_amd"

    def check_runtime(self) -> None:
        if not _is_rocm_runtime():
            raise RuntimeError(
                "triton_amd backend requires PyTorch ROCm runtime (torch.version.hip + CUDA device visibility)."
            )
        if not is_triton_amd_available():
            raise RuntimeError("triton_amd backend is unavailable: Triton HIP target not detected.")

    def _real_dtype(self) -> torch.dtype:
        return torch.float32 if self.precision == "float32" else torch.float64

    def _complex_dtype(self) -> torch.dtype:
        return torch.complex64 if self.precision == "float32" else torch.complex128

    def _to_2d(self, data: Any, *, dtype: torch.dtype) -> torch.Tensor:
        x = torch.as_tensor(data, device=f"cuda:{self.device_id}", dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got {x.ndim}D.")
        return x.contiguous()

    def _allocate_complex(self, batch: int, state_len: int) -> torch.Tensor:
        real_dtype = self._real_dtype()
        out_ri = torch.empty(
            (batch, state_len, 2),
            device=f"cuda:{self.device_id}",
            dtype=real_dtype,
        )
        return torch.view_as_complex(out_ri).to(self._complex_dtype())

    def encode_amplitude(self, data: Any, num_qubits: int) -> torch.Tensor:
        x = self._to_2d(data, dtype=self._real_dtype())
        batch, sample_size = x.shape
        state_len = 1 << num_qubits
        if sample_size != state_len:
            raise ValueError(
                f"Amplitude encoding expects sample size {state_len} (=2^num_qubits), got {sample_size}."
            )
        norms = torch.linalg.vector_norm(x, dim=1).clamp_min(1e-12)
        inv = (1.0 / norms).to(x.dtype)

        out_ri = torch.empty((batch, state_len, 2), device=x.device, dtype=x.dtype)
        total = batch * state_len
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK"]),)
        kernel: Any = _amplitude_pack_kernel[grid]
        kernel(
            x,
            inv,
            out_ri,
            sample_size,
            state_len,
            total,
            BLOCK=256,
        )
        return torch.view_as_complex(out_ri).to(self._complex_dtype())

    def encode_angle(self, data: Any, num_qubits: int) -> torch.Tensor:
        real_dtype = self._real_dtype()
        angles = self._to_2d(data, dtype=real_dtype)
        batch, width = angles.shape
        if width != num_qubits:
            raise ValueError(
                f"Angle encoding expects sample size {num_qubits} (=num_qubits), got {width}."
            )
        state_len = 1 << num_qubits

        out_ri = torch.empty((batch, state_len, 2), device=angles.device, dtype=real_dtype)
        total = batch * state_len
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK"]),)
        kernel: Any = _angle_kernel[grid]
        kernel(
            angles,
            out_ri,
            state_len,
            total,
            NQ=num_qubits,
            FP64=self.precision == "float64",
            BLOCK=128,
        )
        return torch.view_as_complex(out_ri).to(self._complex_dtype())

    def encode_basis(self, data: Any, num_qubits: int) -> torch.Tensor:
        idx = torch.as_tensor(data, device=f"cuda:{self.device_id}", dtype=torch.int64)
        if idx.ndim == 2:
            if idx.shape[1] != 1:
                raise ValueError(f"Basis 2D input expects width 1, got {idx.shape[1]}.")
            idx = idx.squeeze(1)
        elif idx.ndim != 1:
            raise ValueError(f"Expected 1D or 2D basis input, got {idx.ndim}D.")
        if idx.numel() == 0:
            raise ValueError("Basis tensor cannot be empty.")
        state_len = 1 << num_qubits
        if torch.any(idx < 0) or torch.any(idx >= state_len):
            raise ValueError(
                f"Basis index out of range. Valid range is [0, {state_len - 1}]."
            )
        idx = idx.contiguous()
        batch = int(idx.numel())

        real_dtype = self._real_dtype()
        out_ri = torch.empty((batch, state_len, 2), device=idx.device, dtype=real_dtype)
        total = batch * state_len
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK"]),)
        kernel: Any = _basis_kernel[grid]
        kernel(
            idx,
            out_ri,
            state_len,
            total,
            FP64=self.precision == "float64",
            BLOCK=256,
        )
        return torch.view_as_complex(out_ri).to(self._complex_dtype())

    def encode(
        self,
        data: Any,
        num_qubits: int,
        encoding_method: str = "amplitude",
    ) -> Any:
        self.check_runtime()
        method = encoding_method.lower()
        value: torch.Tensor
        if method == "amplitude":
            value = self.encode_amplitude(data, num_qubits)
        elif method == "angle":
            value = self.encode_angle(data, num_qubits)
        elif method == "basis":
            value = self.encode_basis(data, num_qubits)
        else:
            raise ValueError(
                f"Unsupported encoding '{encoding_method}'. triton_amd supports amplitude, angle, basis."
            )

        # Return the same routed contract even when engine is used directly.
        from qumat_qdp.backend import QuantumTensor

        return QuantumTensor(value=value, backend=self.backend)
