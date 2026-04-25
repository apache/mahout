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

import math
from dataclasses import dataclass, field
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
triton_lang = _load_optional_module("triton.language")


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


# ---------------------------------------------------------------------------
# Triton kernel: fused phase encoder (real-only path).
#
# One kernel per program covers BLOCK output basis-states for a single sample,
# fusing: bit-pattern materialization + θ(b) accumulation + sin/cos + 1/√2^n
# scaling + complex-pack into the (B, S) real/imag planes. The PyTorch path
# below allocates 5 intermediates of size O(B · S); this kernel writes the
# output in a single pass.
#
# Real and imag planes are written as separate float buffers, then the caller
# stitches them via ``torch.complex`` (free metadata view; PyTorch fuses the
# stride pattern). This avoids needing complex-typed pointers in Triton, which
# the HIP backend does not support directly.
#
# Limitations: float32 + n_qubits ≤ 32 (single int32 bit packing).  For n > 32
# or float64 the engine falls back to the vectorized PyTorch path, which is
# already memory-bound, not compute-bound.
# ---------------------------------------------------------------------------

if triton_mod is not None and triton_lang is not None:
    tl = triton_lang

    @triton_mod.jit
    def _phase_encode_kernel(
        phases_ptr,  # *fp32, shape (B, n_qubits)
        out_ptr,  # *fp32, view-as-real of complex64 output: (B, 2·state_len)
        n_qubits,
        state_len,
        norm_factor,  # 1/√2^n
        BLOCK: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)

        s_offsets = pid_s * BLOCK + tl.arange(0, BLOCK)
        s_mask = s_offsets < state_len

        # φ(b) = Σ_k phases[k] · ((b >> k) & 1) — fused bit unpack + accumulate.
        phi = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(0, n_qubits):
            bit_k = ((s_offsets >> k) & 1).to(tl.float32)
            phase_k = tl.load(phases_ptr + pid_b * n_qubits + k)
            phi += phase_k * bit_k

        re = tl.cos(phi) * norm_factor
        im = tl.sin(phi) * norm_factor

        # Write interleaved into the complex64 buffer's float view: each
        # output element occupies two adjacent floats (re, im). One kernel,
        # one allocation; no separate planes that would need a final stitch.
        base = pid_b * state_len * 2 + s_offsets * 2
        tl.store(out_ptr + base, re, mask=s_mask)
        tl.store(out_ptr + base + 1, im, mask=s_mask)

else:  # pragma: no cover - non-Triton hosts use the PyTorch fallback
    _phase_encode_kernel = None


# Largest n the ZZ pair-matrix path will materialize before we refuse and
# point the user at the loop fallback. State vector at n=20 is 16 MiB cf64;
# pair matrix at n=20 is 1 MiB · 190 entries · 4 B = ~760 MiB — so this is the
# right cutoff before pair_matrix dominates the AMD HBM budget.
_IQP_PAIR_MATRIX_MAX_N = 20


@dataclass
class TritonAmdEngine:
    """AMD backend implementing amplitude/angle/basis/iqp/iqp-z/phase encoders."""

    device_id: int = 0
    precision: str = "float32"

    # Per-engine cache of (n_qubits → bits table) keyed by (n, real_dtype).
    # Avoids regenerating the (state_len, n_qubits) bit pattern on every call;
    # the table is reused across batches for any encoder that needs it.
    _bits_cache: dict = field(default_factory=dict, repr=False, compare=False)
    # Cache of (n → upper-triangular pair index) for IQP-ZZ.
    _pair_cache: dict = field(default_factory=dict, repr=False, compare=False)

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
        # Fast path: caller already supplies a 2-D, contiguous, on-device,
        # correctly-typed torch tensor (the common case for benchmarks and
        # downstream pipelines). Skip ``as_tensor`` + ``contiguous`` work.
        if (
            isinstance(data, torch_mod.Tensor)
            and data.ndim == 2
            and data.dtype is dtype
            and data.is_contiguous()
            and data.device.type == "cuda"
            and data.device.index == self.device_id
        ):
            return data
        x = torch_mod.as_tensor(data, device=self._device(), dtype=dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got {x.ndim}D.")
        return x.contiguous()

    def _bits_table(self, num_qubits: int, real_dtype: Any) -> Any:
        """Cached ``bits[b, k] = (b >> k) & 1`` table cast to ``real_dtype``.

        Returned shape is ``(2^num_qubits, num_qubits)``. The same table is
        reused by ``encode_angle``/``encode_iqp``/``encode_phase`` across
        successive batches at the same ``num_qubits``.
        """
        torch_mod = self._require_torch()
        key = (num_qubits, real_dtype)
        cached = self._bits_cache.get(key)
        if cached is not None:
            return cached
        device = torch_mod.device(self._device())
        state_len = 1 << num_qubits
        b_idx = torch_mod.arange(state_len, device=device, dtype=torch_mod.int64)
        k_idx = torch_mod.arange(num_qubits, device=device, dtype=torch_mod.int64)
        bits = ((b_idx.unsqueeze(1) >> k_idx) & 1).to(real_dtype).contiguous()
        self._bits_cache[key] = bits
        return bits

    def _pair_indices(self, num_qubits: int) -> Any:
        """Cached ``(n*(n-1)/2, 2)`` table of upper-triangular qubit pairs."""
        torch_mod = self._require_torch()
        cached = self._pair_cache.get(num_qubits)
        if cached is not None:
            return cached
        device = torch_mod.device(self._device())
        pairs = torch_mod.combinations(torch_mod.arange(num_qubits, device=device), r=2)
        self._pair_cache[num_qubits] = pairs
        return pairs

    def encode_amplitude(self, data: Any, num_qubits: int) -> Any:
        torch_mod = self._require_torch()
        x = self._to_2d(data, dtype=self._real_dtype())
        batch, sample_size = x.shape
        state_len = 1 << num_qubits
        if sample_size > state_len:
            raise ValueError(
                f"Amplitude encoding expects sample size <= {state_len} (=2^num_qubits), got {sample_size}."
            )

        norms = torch_mod.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)
        amp = x / norms
        if sample_size < state_len:
            # F.pad is a single fused op vs a separate zeros + cat.
            amp = torch_mod.nn.functional.pad(amp, (0, state_len - sample_size))
        # ``.to(complex_dtype)`` from a real tensor is one kernel that writes
        # (real, 0) interleaved — strictly better than building a separate
        # zeros tensor and combining via ``torch.complex(real, zeros)``.
        return amp.to(self._complex_dtype())

    def encode_angle(self, data: Any, num_qubits: int) -> Any:
        torch_mod = self._require_torch()
        real_dtype = self._real_dtype()
        angles = self._to_2d(data, dtype=real_dtype)
        batch, width = angles.shape
        if width != num_qubits:
            raise ValueError(
                f"Angle encoding expects sample size {num_qubits} (=num_qubits), got {width}."
            )

        bits = self._bits_table(num_qubits, real_dtype)  # (S, n) cached

        # amp[batch, b] = prod_k (sin(θ_k) if bit_k else cos(θ_k))
        # Closed-form vectorization: broadcast (B, 1, n) sin/cos against
        # (1, S, n) bit pattern, gather via where, reduce-product over k.
        # One allocation for the (B, S, n) workspace; the previous Python-level
        # n-step loop allocated a fresh (B, S) tensor per iteration.
        sin = torch_mod.sin(angles).unsqueeze(1)
        cos = torch_mod.cos(angles).unsqueeze(1)
        factor = torch_mod.where(bits.unsqueeze(0) > 0.5, sin, cos)
        amp = factor.prod(dim=2)
        return amp.to(self._complex_dtype())

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
        complex_dtype = self._complex_dtype()
        out = torch_mod.zeros(
            (batch, state_len),
            device=idx.device,
            dtype=complex_dtype,
        )
        out.scatter_(
            1,
            idx.reshape(batch, 1),
            torch_mod.ones((batch, 1), device=idx.device, dtype=complex_dtype),
        )
        return out

    def _iqp_phase(
        self,
        params: Any,
        num_qubits: int,
        bits: Any,
        *,
        enable_zz: bool,
    ) -> Any:
        """Compute θ(x) = Σ x_i·data_i (+ Σ_{i<j} x_i x_j data_ij if ZZ).

        Returns shape ``(batch, 2**num_qubits)`` in the real dtype.
        """
        torch_mod = self._require_torch()
        n = num_qubits
        z_params = params[:, :n]
        # phase = z_params @ bits.T : (B, S)
        phase = torch_mod.matmul(z_params, bits.T)
        if enable_zz and n >= 2:
            if n > _IQP_PAIR_MATRIX_MAX_N:
                # Pair matrix is (S, n_pairs) — at n=20 that's already ~760 MiB
                # in float32. Past this size, fall back to a per-pair loop.
                # Slower but bounded memory; the workload itself is also
                # impractical at this point (state vector alone is multi-GB).
                pair_idx = n
                zz_params = params
                for i in range(n - 1):
                    bi = bits[:, i]
                    for j in range(i + 1, n):
                        bj = bits[:, j]
                        phase = phase + zz_params[:, pair_idx : pair_idx + 1] * (
                            bi * bj
                        ).unsqueeze(0)
                        pair_idx += 1
            else:
                zz_params = params[:, n:]
                pairs = self._pair_indices(n)
                pair_matrix = bits[:, pairs[:, 0]] * bits[:, pairs[:, 1]]
                phase = phase + torch_mod.matmul(zz_params, pair_matrix.T)
        return phase

    def encode_iqp(
        self,
        data: Any,
        num_qubits: int,
        *,
        enable_zz: bool = True,
    ) -> Any:
        torch_mod = self._require_torch()
        real_dtype = self._real_dtype()
        params = self._to_2d(data, dtype=real_dtype)
        batch, width = params.shape

        n = num_qubits
        expected = n + n * (n - 1) // 2 if enable_zz else n
        if width != expected:
            variant = "ZZ" if enable_zz else "Z-only"
            raise ValueError(
                f"IQP encoding ({variant}) expects {expected} parameters for {n} qubits, got {width}."
            )

        state_len = 1 << n
        bits = self._bits_table(n, real_dtype)
        phase = self._iqp_phase(params, n, bits, enable_zz=enable_zz)

        # f[x] = exp(i·θ(x)). ``torch.complex(cos, sin)`` allocates a single
        # contiguous complex tensor and is faster than writing into strided
        # ``.real``/``.imag`` views of a separately-allocated complex buffer.
        f = torch_mod.complex(torch_mod.cos(phase), torch_mod.sin(phase)).to(
            self._complex_dtype()
        )

        # In-place n-stage Walsh-Hadamard butterfly. View ``f`` as
        # (B, K, 2, stride) per stage and do (a, b) ← (a + b, a - b) using a
        # single ``state_len/2``-sized scratch buffer instead of allocating
        # two (lo+hi, lo-hi) buffers and concatenating them every stage.
        if n > 0:
            scratch = torch_mod.empty(
                (batch, state_len // 2), device=f.device, dtype=f.dtype
            )
            for s in range(n):
                stride = 1 << s
                view = f.view(batch, state_len // (stride * 2), 2, stride)
                a = view.select(2, 0)
                b = view.select(2, 1)
                scratch_view = scratch.view(batch, state_len // (stride * 2), stride)
                torch_mod.sub(a, b, out=scratch_view)  # scratch ← a − b
                a.add_(b)  # a ← a + b (in-place)
                b.copy_(scratch_view)  # b ← (a − b) from scratch
            f = f.view(batch, state_len)

        f.mul_(1.0 / float(state_len))
        return f

    def _can_use_triton_phase_kernel(self, num_qubits: int) -> bool:
        return (
            _phase_encode_kernel is not None
            and self.precision == "float32"
            and 1 <= num_qubits <= 32
        )

    def _encode_phase_triton(self, phases: Any, num_qubits: int) -> Any:
        """Triton-fused phase encoder for float32 / n ≤ 32.

        One HIP kernel launch per (sample, output-tile) pair; fuses the
        bit-table materialization + θ(b) accumulate + cos/sin + 1/√2^n scale
        + complex-pack into a single pass that writes the output buffer
        interleaved (re, im, re, im, …) — the native complex64 layout.
        """
        torch_mod = self._require_torch()
        # ``_can_use_triton_phase_kernel`` already guards on Triton being
        # available; this assertion narrows the type for the type checker.
        assert _phase_encode_kernel is not None
        batch = phases.shape[0]
        state_len = 1 << num_qubits

        # Allocate the complex output once; pass its real-view as a flat
        # (B, 2·S) float32 buffer to the kernel for direct interleaved writes.
        out = torch_mod.empty(
            (batch, state_len),
            device=phases.device,
            dtype=torch_mod.complex64,
        )
        out_real_view = torch_mod.view_as_real(out).view(batch, state_len * 2)

        norm = math.pow(math.sqrt(0.5), num_qubits)
        BLOCK = 256
        grid = (batch, (state_len + BLOCK - 1) // BLOCK)
        _phase_encode_kernel[grid](
            phases,
            out_real_view,
            num_qubits,
            state_len,
            norm,
            BLOCK=BLOCK,
        )
        return out

    def encode_phase(self, data: Any, num_qubits: int) -> Any:
        torch_mod = self._require_torch()
        real_dtype = self._real_dtype()
        phases = self._to_2d(data, dtype=real_dtype)
        batch, width = phases.shape
        if width != num_qubits:
            raise ValueError(
                f"Phase encoding expects sample size {num_qubits} (=num_qubits), got {width}."
            )

        if self._can_use_triton_phase_kernel(num_qubits):
            return self._encode_phase_triton(phases, num_qubits)

        # Fallback: vectorized PyTorch path (float64 or n > 32).
        bits = self._bits_table(num_qubits, real_dtype)
        phi = torch_mod.matmul(phases, bits.T)
        norm = math.pow(math.sqrt(0.5), num_qubits)
        # ``torch.complex(re, im)`` writes a contiguous interleaved buffer in
        # one allocation — faster than ``empty(complex)`` followed by strided
        # writes into ``.real``/``.imag``.
        return torch_mod.complex(
            torch_mod.cos(phi).mul_(norm),
            torch_mod.sin(phi).mul_(norm),
        ).to(self._complex_dtype())

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
        if method == "iqp":
            return self.encode_iqp(data, num_qubits, enable_zz=True)
        if method == "iqp-z":
            return self.encode_iqp(data, num_qubits, enable_zz=False)
        if method == "phase":
            return self.encode_phase(data, num_qubits)
        raise ValueError(
            f"Unsupported encoding '{encoding_method}'. "
            "triton_amd supports amplitude, angle, basis, iqp, iqp-z, phase."
        )
