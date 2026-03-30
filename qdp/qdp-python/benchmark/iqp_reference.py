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

"""Torch reference implementation for IQP encoding.

The reference mirrors the CUDA kernel semantics:

- build the phase vector from Z and optional ZZ terms
- apply an unnormalized Walsh-Hadamard transform
- divide by ``2**num_qubits`` at the end

The helpers here are intentionally explicit rather than clever so they can serve
as a correctness oracle and a ``torch.compile`` baseline in benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

IQP_ENCODING_METHODS = ("iqp", "iqp-z")


def iqp_enable_zz(encoding_method: str) -> bool:
    """Return whether the requested IQP variant includes ZZ interaction terms."""
    normalized = encoding_method.lower()
    if normalized not in IQP_ENCODING_METHODS:
        raise ValueError(
            f"Unsupported IQP encoding method '{encoding_method}'. "
            f"Expected one of: {', '.join(IQP_ENCODING_METHODS)}"
        )
    return normalized == "iqp"


def iqp_full_data_len(num_qubits: int) -> int:
    """Return the number of parameters for full IQP encoding."""
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")
    return num_qubits + num_qubits * max(num_qubits - 1, 0) // 2


def iqp_z_data_len(num_qubits: int) -> int:
    """Return the number of parameters for IQP-Z encoding."""
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")
    return num_qubits


def iqp_sample_size(num_qubits: int, enable_zz: bool = True) -> int:
    """Return the expected per-sample parameter count for the requested variant."""
    return iqp_full_data_len(num_qubits) if enable_zz else iqp_z_data_len(num_qubits)


def iqp_sample_size_for_method(num_qubits: int, encoding_method: str) -> int:
    """Return the expected per-sample parameter count for an IQP method name."""
    return iqp_sample_size(num_qubits, enable_zz=iqp_enable_zz(encoding_method))


def _build_feature_matrices(
    num_qubits: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    state_len = 1 << num_qubits
    basis = torch.arange(state_len, device=device, dtype=torch.int64)
    qubits = torch.arange(num_qubits, device=device, dtype=torch.int64)
    bits = ((basis[:, None] >> qubits) & 1).to(dtype=dtype)

    if num_qubits < 2:
        return bits, None

    pair_idx = torch.triu_indices(num_qubits, num_qubits, offset=1, device=device)
    pair_terms = bits[:, pair_idx[0]] * bits[:, pair_idx[1]]
    return bits, pair_terms.to(dtype=dtype)


def _fwht_last_dim(state: torch.Tensor) -> torch.Tensor:
    """Apply an in-place-equivalent FWHT along the last dimension."""
    width = state.shape[-1]
    stage = 1
    while stage < width:
        blocks = width // (stage * 2)
        reshaped = state.reshape(*state.shape[:-1], blocks, 2, stage)
        left = reshaped[..., 0, :]
        right = reshaped[..., 1, :]
        state = torch.stack((left + right, left - right), dim=-2).reshape(
            *state.shape[:-1], width
        )
        stage <<= 1
    return state


def build_iqp_reference(
    num_qubits: int,
    *,
    enable_zz: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
    compile_reference: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a reusable torch IQP reference for a fixed qubit count.

    The returned callable accepts a 1D sample tensor or a 2D batch tensor and
    returns a 2D complex state tensor with shape ``[batch_size, 2**num_qubits]``.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    basis_terms, pair_terms = _build_feature_matrices(num_qubits, target_device, dtype)
    expected_len = iqp_sample_size(num_qubits, enable_zz)
    state_len = 1 << num_qubits

    def reference(data: torch.Tensor) -> torch.Tensor:
        if data.dim() == 1:
            batch = data.unsqueeze(0)
        elif data.dim() == 2:
            batch = data
        else:
            raise ValueError(
                f"IQP reference expects a 1D or 2D tensor, got {data.dim()}D"
            )

        if batch.shape[-1] != expected_len:
            raise ValueError(
                f"IQP reference expects {expected_len} parameters for {num_qubits} qubits, "
                f"got {batch.shape[-1]}"
            )

        batch = batch.to(device=target_device, dtype=dtype)
        phase = batch[:, :num_qubits] @ basis_terms.T
        if enable_zz and pair_terms is not None:
            phase = phase + batch[:, num_qubits:] @ pair_terms.T

        state = torch.polar(torch.ones_like(phase), phase)
        state = _fwht_last_dim(state) / state_len
        return state

    if compile_reference and hasattr(torch, "compile"):
        try:
            reference = torch.compile(reference, mode="reduce-overhead", fullgraph=True)
        except Exception:
            pass

    return reference


def iqp_reference_torch(
    data: torch.Tensor | list[float] | tuple[float, ...],
    num_qubits: int,
    *,
    enable_zz: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
    compile_reference: bool = False,
) -> torch.Tensor:
    """Convenience wrapper around :func:`build_iqp_reference`."""
    if isinstance(data, torch.Tensor):
        tensor = data.to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(data, device=device, dtype=dtype)
    reference = build_iqp_reference(
        num_qubits,
        enable_zz=enable_zz,
        device=tensor.device,
        dtype=tensor.dtype,
        compile_reference=compile_reference,
    )
    return reference(tensor)


@dataclass(frozen=True)
class IqpReferenceResult:
    """Small wrapper used by benchmarks when timing multiple implementations."""

    name: str
    duration_sec: float
    latency_ms_per_vector: float


__all__ = [
    "IQP_ENCODING_METHODS",
    "IqpReferenceResult",
    "build_iqp_reference",
    "iqp_enable_zz",
    "iqp_full_data_len",
    "iqp_reference_torch",
    "iqp_sample_size",
    "iqp_sample_size_for_method",
    "iqp_z_data_len",
]
