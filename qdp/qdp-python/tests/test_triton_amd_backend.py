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

import pytest
import torch
from qumat_qdp import QdpEngine, is_triton_amd_available
from qumat_qdp.triton_amd import TritonAmdEngine


def _as_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.from_dlpack(value)


def _torch_amplitude_ref(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    norms = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)
    y = x / norms
    return torch.complex(y, torch.zeros_like(y))


def _torch_angle_ref(angles: torch.Tensor, num_qubits: int) -> torch.Tensor:
    real_dtype = angles.dtype
    batch = angles.shape[0]
    state_len = 1 << num_qubits
    idx = torch.arange(state_len, device=angles.device).reshape(1, state_len)
    amp = torch.ones((batch, state_len), device=angles.device, dtype=real_dtype)
    for bit in range(num_qubits):
        col = angles[:, bit].unsqueeze(1)
        factor = torch.where(
            ((idx >> bit) & 1) == 1,
            torch.sin(col),
            torch.cos(col),
        )
        amp = amp * factor
    return torch.complex(amp, torch.zeros_like(amp))


def _torch_basis_ref(idx: torch.Tensor, num_qubits: int) -> torch.Tensor:
    idx = idx.to(torch.int64)
    batch = idx.numel()
    state_len = 1 << num_qubits
    out = torch.zeros((batch, state_len), device=idx.device, dtype=torch.complex64)
    out.scatter_(
        1,
        idx.reshape(batch, 1),
        torch.ones((batch, 1), device=idx.device, dtype=torch.complex64),
    )
    return out


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_amplitude_parity() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float32")
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    got = _as_torch(engine.encode(x, 3, "amplitude"))
    ref = _torch_amplitude_ref(x)
    assert got.shape == ref.shape
    assert got.dtype == torch.complex64
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_amplitude_pad_parity() -> None:
    """Sample sizes < 2**num_qubits zero-pad to the state length, matching the CUDA path."""
    engine = TritonAmdEngine(device_id=0, precision="float32")
    x = torch.randn(4, 6, device="cuda", dtype=torch.float32)
    got = _as_torch(engine.encode(x, 3, "amplitude"))
    assert got.shape == (4, 8)
    assert got.dtype == torch.complex64
    norms = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)
    expected_real = torch.cat([x / norms, torch.zeros((4, 2), device=x.device)], dim=1)
    assert torch.allclose(got.real, expected_real, atol=1e-5, rtol=1e-5)
    assert torch.allclose(got.imag, torch.zeros_like(got.imag), atol=0.0)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_amplitude_oversize_rejected() -> None:
    """Sample sizes > 2**num_qubits remain a hard error (matches CUDA contract)."""
    engine = TritonAmdEngine(device_id=0, precision="float32")
    x = torch.randn(2, 9, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="<= 8"):
        engine.encode(x, 3, "amplitude")


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_angle_parity() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float32")
    angles = torch.randn(3, 5, device="cuda", dtype=torch.float32)
    got = _as_torch(engine.encode(angles, 5, "angle"))
    ref = _torch_angle_ref(angles, 5)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_basis_parity() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float32")
    idx = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int64)
    got = _as_torch(engine.encode(idx, 3, "basis"))
    ref = _torch_basis_ref(idx, 3)
    assert got.shape == ref.shape
    assert torch.equal(got, ref)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_angle_float64_precision_contract() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float64")
    angles = torch.randn(2, 4, device="cuda", dtype=torch.float64)
    got = _as_torch(engine.encode(angles, 4, "angle"))
    ref = _torch_angle_ref(angles.to(torch.float64), 4).to(torch.complex128)
    assert got.dtype == torch.complex128
    assert torch.allclose(got, ref, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch.version, "cuda", None) is None,
    reason="NVIDIA CUDA reference not available",
)
@pytest.mark.rocm
def test_triton_amd_cuda_reference_optional() -> None:
    _qdp = pytest.importorskip("_qdp")
    if not is_triton_amd_available():
        pytest.skip("Triton AMD backend unavailable")

    engine_triton = TritonAmdEngine(device_id=0, precision="float32")
    engine_cuda = _qdp.QdpEngine(0, precision="float32")
    x = torch.randn(2, 4, device="cuda", dtype=torch.float32)
    got = _as_torch(engine_triton.encode(x, 2, "amplitude"))
    ref = torch.from_dlpack(engine_cuda.encode(x, 2, "amplitude"))
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_direct_engine_returns_dlpack_contract() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float32")
    x = torch.randn(2, 4, device="cuda", dtype=torch.float32)
    qt = engine.encode(x, 2, "amplitude")
    assert hasattr(qt, "__dlpack__")
    out = torch.from_dlpack(qt)
    assert out.shape == (2, 4)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_unified_router_contract_returns_torch_tensor() -> None:
    router = QdpEngine(backend="amd", device_id=0, precision="float32")
    x = torch.randn(2, 4, device="cuda", dtype=torch.float32)
    qt = router.encode(x, 2, "amplitude")
    assert isinstance(qt, torch.Tensor)
    assert qt.shape == (2, 4)
    assert qt.dtype == torch.complex64
