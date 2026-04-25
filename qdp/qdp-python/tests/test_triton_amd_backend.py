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

import math

import pytest
import torch
from qumat_qdp import QdpEngine, is_triton_amd_available
from qumat_qdp.torch_ref import iqp_encode as _torch_ref_iqp
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


def _torch_phase_ref(phases: torch.Tensor, num_qubits: int) -> torch.Tensor:
    real_dtype = phases.dtype
    batch = phases.shape[0]
    state_len = 1 << num_qubits
    idx = torch.arange(state_len, device=phases.device, dtype=torch.int64)
    bits = (
        (idx.unsqueeze(1) >> torch.arange(num_qubits, device=phases.device)) & 1
    ).to(real_dtype)
    phi = phases @ bits.T
    norm = math.pow(math.sqrt(0.5), num_qubits)
    out = torch.complex(torch.cos(phi) * norm, torch.sin(phi) * norm)
    assert out.shape == (batch, state_len)
    return out


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


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_iqp_full_parity_with_torch_ref() -> None:
    n = 4
    engine = TritonAmdEngine(device_id=0, precision="float32")
    data = torch.randn(3, n + n * (n - 1) // 2, device="cuda", dtype=torch.float32)
    got = _as_torch(engine.encode(data, n, "iqp"))
    ref = _torch_ref_iqp(data, n, enable_zz=True)
    assert got.shape == ref.shape
    assert got.dtype == torch.complex64
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_iqp_z_only_parity_with_torch_ref() -> None:
    n = 5
    engine = TritonAmdEngine(device_id=0, precision="float32")
    data = torch.randn(2, n, device="cuda", dtype=torch.float32)
    got = _as_torch(engine.encode(data, n, "iqp-z"))
    ref = _torch_ref_iqp(data, n, enable_zz=False)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_iqp_param_count_validation() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float32")
    # ZZ variant for n=4 expects 4 + 6 = 10 params; pass 9.
    bad = torch.randn(2, 9, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="expects 10 parameters"):
        engine.encode(bad, 4, "iqp")
    # Z-only variant for n=4 expects 4 params; pass 5.
    bad_z = torch.randn(2, 5, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="expects 4 parameters"):
        engine.encode(bad_z, 4, "iqp-z")


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_iqp_normalization_unit_norm() -> None:
    """IQP output is a normalized state vector: Σ|amp|² ≈ 1."""
    engine = TritonAmdEngine(device_id=0, precision="float32")
    n = 6
    data = torch.randn(4, n + n * (n - 1) // 2, device="cuda", dtype=torch.float32)
    got = _as_torch(engine.encode(data, n, "iqp"))
    norms = (got.abs() ** 2).sum(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_phase_parity() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float32")
    phases = torch.randn(3, 5, device="cuda", dtype=torch.float32)
    got = _as_torch(engine.encode(phases, 5, "phase"))
    ref = _torch_phase_ref(phases, 5)
    assert got.shape == ref.shape
    assert got.dtype == torch.complex64
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_phase_normalization_unit_norm() -> None:
    """Phase output is a uniform-magnitude product state: Σ|amp|² ≈ 1."""
    engine = TritonAmdEngine(device_id=0, precision="float32")
    n = 6
    phases = torch.randn(4, n, device="cuda", dtype=torch.float32)
    got = _as_torch(engine.encode(phases, n, "phase"))
    norms = (got.abs() ** 2).sum(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_phase_param_count_validation() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float32")
    bad = torch.randn(2, 3, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="sample size 4"):
        engine.encode(bad, 4, "phase")


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_phase_float64_precision_contract() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float64")
    phases = torch.randn(2, 4, device="cuda", dtype=torch.float64)
    got = _as_torch(engine.encode(phases, 4, "phase"))
    ref = _torch_phase_ref(phases, 4).to(torch.complex128)
    assert got.dtype == torch.complex128
    assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_iqp_float64_precision_contract() -> None:
    """Float64 IQP matches torch_ref bit-close (covers the dtype contract)."""
    engine = TritonAmdEngine(device_id=0, precision="float64")
    n = 4
    data = torch.randn(3, n + n * (n - 1) // 2, device="cuda", dtype=torch.float64)
    got = _as_torch(engine.encode(data, n, "iqp"))
    ref = _torch_ref_iqp(data, n, enable_zz=True).to(torch.complex128)
    assert got.dtype == torch.complex128
    assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_triton_amd_unsupported_method_message_lists_all() -> None:
    engine = TritonAmdEngine(device_id=0, precision="float32")
    with pytest.raises(ValueError) as excinfo:
        engine.encode(torch.zeros(1, 4, device="cuda"), 2, "no-such-method")
    msg = str(excinfo.value)
    for name in ("amplitude", "angle", "basis", "iqp", "iqp-z", "phase"):
        assert name in msg


@pytest.mark.skipif(
    not is_triton_amd_available(), reason="Triton AMD backend unavailable"
)
@pytest.mark.rocm
def test_unified_router_iqp_and_phase_routes() -> None:
    """The public QdpEngine(backend='amd') router accepts iqp/iqp-z/phase too."""
    router = QdpEngine(backend="amd", device_id=0, precision="float32")
    n = 3
    data_iqp = torch.randn(2, n + n * (n - 1) // 2, device="cuda", dtype=torch.float32)
    qt = router.encode(data_iqp, n, "iqp")
    assert isinstance(qt, torch.Tensor)
    assert qt.shape == (2, 1 << n)
    qt_z = router.encode(torch.randn(2, n, device="cuda"), n, "iqp-z")
    assert qt_z.shape == (2, 1 << n)
    qt_p = router.encode(torch.randn(2, n, device="cuda"), n, "phase")
    assert qt_p.shape == (2, 1 << n)
