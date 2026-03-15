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

"""
Pure-PyTorch reference implementations of QDP quantum encoding methods.

Serves two purposes:
1. Speed benchmark reference (compare QDP Rust+CUDA vs PyTorch GPU).
2. Fallback when the ``_qdp`` Rust extension is unavailable.

All functions are fully vectorized (no Python loops over batch/state dims)
and return complex tensors of shape ``(batch_size, 2**num_qubits)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["amplitude_encode", "angle_encode", "basis_encode", "encode", "iqp_encode"]

_COMPLEX_DTYPE_MAP = {
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Map a real dtype to its complex counterpart."""
    result = _COMPLEX_DTYPE_MAP.get(real_dtype)
    if result is None:
        raise TypeError(
            f"Unsupported dtype {real_dtype} for complex conversion. "
            "Use float32 or float64."
        )
    return result


def _ensure_2d(data: torch.Tensor) -> torch.Tensor:
    """Promote 1-D input to ``(1, features)``."""
    if data.ndim == 1:
        return data.unsqueeze(0)
    if data.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D tensor, got {data.ndim}-D")
    return data


def _check_float_dtype(data: torch.Tensor) -> None:
    """Check that *data* is floating-point (dtype only, no GPU sync)."""
    if not data.is_floating_point():
        raise ValueError(f"Expected floating-point input, got {data.dtype}")


# ---------------------------------------------------------------------------
# Amplitude encoding
# ---------------------------------------------------------------------------


def amplitude_encode(
    data: torch.Tensor,
    num_qubits: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Amplitude encoding: L2-normalize and zero-pad to ``2**num_qubits``.

    Args:
        data: Real tensor of shape ``(batch, features)`` or ``(features,)``.
        num_qubits: Number of qubits; state dimension is ``2**num_qubits``.
        device: Target device. Defaults to the device of *data*.

    Returns:
        Complex tensor of shape ``(batch, 2**num_qubits)``.
    """
    data = _ensure_2d(data)
    if device is not None:
        data = data.to(device=torch.device(device))

    _check_float_dtype(data)

    state_dim = 1 << num_qubits
    features = data.shape[1]
    if features > state_dim:
        raise ValueError(
            f"Input features ({features}) exceed state dimension ({state_dim})"
        )

    # Zero-pad to state_dim if needed.
    if features < state_dim:
        data = F.pad(data, (0, state_dim - features))

    # L2-normalize per row.  Clamp avoids GPU→CPU sync from checking for
    # zero norms; a zero-norm row produces near-zero output instead of
    # crashing (NaN/Inf propagate naturally through downstream ops).
    norms = torch.linalg.vector_norm(data, dim=1, keepdim=True)
    data = data / norms.clamp(min=1e-10)

    # Cast to complex (real-to-complex sets imaginary part to zero).
    return data.to(_complex_dtype(data.dtype))


# ---------------------------------------------------------------------------
# Angle encoding
# ---------------------------------------------------------------------------


def angle_encode(
    data: torch.Tensor,
    num_qubits: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Angle encoding: tensor product of per-qubit Ry rotations.

    For state index *i* with binary representation b_{n-1}...b_0::

        amplitude_i = prod_k (sin(θ_k) if b_k == 1 else cos(θ_k))

    Args:
        data: Real tensor of shape ``(batch, num_qubits)`` or ``(num_qubits,)``
              containing rotation angles.
        num_qubits: Number of qubits.
        device: Target device.

    Returns:
        Complex tensor of shape ``(batch, 2**num_qubits)``.
    """
    data = _ensure_2d(data)
    if device is not None:
        data = data.to(device=torch.device(device))

    _check_float_dtype(data)

    if data.shape[1] != num_qubits:
        raise RuntimeError(
            f"Angle encoding expects {num_qubits} values per sample, got {data.shape[1]}"
        )

    state_dim = 1 << num_qubits

    # Trigonometric values: (batch, num_qubits)
    cos_vals = torch.cos(data)
    sin_vals = torch.sin(data)

    # Bit-pattern matrix: (state_dim, num_qubits)
    # bits[i, k] = (i >> k) & 1
    indices = torch.arange(state_dim, device=data.device, dtype=torch.long)
    bits = (
        (indices.unsqueeze(1) >> torch.arange(num_qubits, device=data.device)) & 1
    ).to(data.dtype)

    # For each state index: amplitude = prod_k (sin if bit else cos)
    # Shape: (batch, state_dim, num_qubits) via broadcasting
    trig = bits.unsqueeze(0) * sin_vals.unsqueeze(1) + (
        1 - bits.unsqueeze(0)
    ) * cos_vals.unsqueeze(1)

    # Product over qubits → (batch, state_dim)
    amplitudes = trig.prod(dim=2)

    return amplitudes.to(_complex_dtype(data.dtype))


# ---------------------------------------------------------------------------
# Basis encoding
# ---------------------------------------------------------------------------


def basis_encode(
    data: torch.Tensor,
    num_qubits: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Basis encoding: one-hot computational basis state.

    Args:
        data: Tensor of integer indices (as float) with shape ``(batch,)``
              or ``(batch, 1)``.
        num_qubits: Number of qubits; state dimension is ``2**num_qubits``.
        device: Target device.

    Returns:
        Complex tensor of shape ``(batch, 2**num_qubits)``.
    """
    if device is not None:
        data = data.to(device=torch.device(device))

    if data.ndim > 2 or (data.ndim == 2 and data.shape[1] != 1):
        raise ValueError(
            f"Basis encoding expects shape (batch,) or (batch, 1), got {data.shape}"
        )

    data = data.flatten()
    batch = data.shape[0]
    state_dim = 1 << num_qubits

    # Validate: finite and integer-valued.
    if data.is_floating_point():
        if torch.any(~torch.isfinite(data)):
            raise RuntimeError("Basis encoding indices must be finite")
        if torch.any(data != data.floor()):
            raise RuntimeError("Basis encoding requires integer-valued indices")
    indices = data.long()

    # Validate range.
    if torch.any(indices < 0):
        raise RuntimeError("Basis encoding indices must be non-negative")
    if torch.any(indices >= state_dim):
        raise RuntimeError(
            f"Basis index exceeds state vector size (2**{num_qubits} = {state_dim})"
        )

    cdtype = (
        _complex_dtype(data.dtype) if data.is_floating_point() else torch.complex128
    )
    result = torch.zeros(batch, state_dim, dtype=cdtype, device=data.device)
    result.scatter_(1, indices.unsqueeze(1), 1.0)
    return result


# ---------------------------------------------------------------------------
# IQP encoding
# ---------------------------------------------------------------------------


def iqp_encode(
    data: torch.Tensor,
    num_qubits: int,
    *,
    device: torch.device | str | None = None,
    enable_zz: bool = True,
) -> torch.Tensor:
    """IQP (Instantaneous Quantum Polynomial) encoding.

    Implements ``|ψ⟩ = H^⊗n · U_phase(data) · H^⊗n |0⟩^⊗n`` using the
    Fast Walsh-Hadamard Transform.

    Args:
        data: Real tensor of shape ``(batch, params)`` or ``(params,)``.
              For *enable_zz=True*: params = n + n*(n-1)/2.
              For *enable_zz=False*: params = n.
        num_qubits: Number of qubits.
        device: Target device.
        enable_zz: If True, include two-qubit ZZ interaction terms.

    Returns:
        Complex tensor of shape ``(batch, 2**num_qubits)``.
    """
    data = _ensure_2d(data)
    if device is not None:
        data = data.to(device=torch.device(device))

    _check_float_dtype(data)

    n = num_qubits
    expected_params = n + n * (n - 1) // 2 if enable_zz else n
    if data.shape[1] != expected_params:
        raise RuntimeError(
            f"IQP encoding ({'ZZ' if enable_zz else 'Z-only'}) expects {expected_params} "
            f"parameters for {n} qubits, got {data.shape[1]}"
        )

    state_dim = 1 << n
    batch = data.shape[0]

    # Build bit patterns for all basis states: (state_dim, n)
    x_indices = torch.arange(state_dim, device=data.device, dtype=torch.long)
    x_bits = ((x_indices.unsqueeze(1) >> torch.arange(n, device=data.device)) & 1).to(
        data.dtype
    )

    # Phase computation for each basis state x:
    # θ(x) = Σ_i x_i * data[i]  (+ ZZ terms if enabled)
    z_params = data[:, :n]  # (batch, n)
    phase = torch.matmul(z_params, x_bits.T)  # (batch, state_dim)

    if enable_zz and n >= 2:
        # Two-qubit ZZ terms: Σ_{i<j} x_i * x_j * data[n + pair_index]
        zz_params = data[:, n:]  # (batch, n*(n-1)//2)
        # Vectorized pair product via torch.combinations
        pairs = torch.combinations(torch.arange(n, device=data.device), r=2)
        pair_matrix = (
            x_bits[:, pairs[:, 0]] * x_bits[:, pairs[:, 1]]
        )  # (state_dim, n_pairs)
        phase = phase + torch.matmul(zz_params, pair_matrix.T)

    # f[x] = exp(i * θ(x))  — use complex tensor for WHT
    f = torch.complex(torch.cos(phase), torch.sin(phase))

    # Fast Walsh-Hadamard Transform: n sequential butterfly stages
    for s in range(n):
        stride = 1 << s
        block = 1 << (s + 1)
        f = f.view(batch, state_dim // block, block)
        lo = f[:, :, :stride]
        hi = f[:, :, stride:]
        f = torch.cat([lo + hi, lo - hi], dim=2)
    f = f.view(batch, state_dim)

    # Normalize by 1/2^n
    f = f * (1.0 / state_dim)

    cdtype = _complex_dtype(data.dtype)
    return f.to(cdtype)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ENCODERS = {
    "amplitude": amplitude_encode,
    "angle": angle_encode,
    "basis": basis_encode,
    "iqp": iqp_encode,
}


def encode(
    data: torch.Tensor,
    num_qubits: int,
    encoding_method: str = "amplitude",
    *,
    device: torch.device | str | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Dispatch to the appropriate encoding function by method name.

    Args:
        data: Input tensor.
        num_qubits: Number of qubits.
        encoding_method: One of ``"amplitude"``, ``"angle"``, ``"basis"``, ``"iqp"``.
        device: Target device.
        **kwargs: Extra arguments forwarded to the encoder (e.g. *enable_zz* for IQP).

    Returns:
        Complex tensor of shape ``(batch, 2**num_qubits)``.
    """
    fn = _ENCODERS.get(encoding_method)
    if fn is None:
        raise ValueError(
            f"Unknown encoding method {encoding_method!r}. "
            f"Supported: {', '.join(sorted(_ENCODERS))}"
        )
    return fn(data, num_qubits, device=device, **kwargs)
