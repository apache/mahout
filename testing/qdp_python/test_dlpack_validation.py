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

"""Unit tests for DLPack validation in encode_from_pytorch (extract_dlpack_tensor)."""

import pytest

torch = pytest.importorskip("torch")
_qdp = pytest.importorskip("_qdp")
QdpEngine = _qdp.QdpEngine


def _cuda_available():
    return torch.cuda.is_available()


def _engine():
    return QdpEngine(0)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_float32_amplitude_supported() -> None:
    """1D float32 CUDA tensor should be supported for amplitude encoding via GPU pointer f32 path."""
    engine = _engine()
    # 1D float32 CUDA tensor (contiguous)
    t = torch.randn(4, dtype=torch.float32, device="cuda")
    result = engine.encode(t, num_qubits=2, encoding_method="amplitude")
    assert result is not None

    # Verify DLPack round-trip works and tensor is on CUDA
    qt = torch.from_dlpack(result)
    assert qt.is_cuda
    # With default engine precision=float32, complex64 is expected
    assert qt.dtype in (torch.complex64, torch.complex128)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_float32_amplitude_2d_supported() -> None:
    """2D float32 CUDA tensor should use the batch GPU-pointer float32 amplitude path."""
    engine = _engine()
    t = torch.tensor(
        [[3.0, 4.0, 0.0, 0.0], [1.0, 2.0, 2.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )

    result = engine.encode(t, num_qubits=2, encoding_method="amplitude")
    assert result is not None

    qt = torch.from_dlpack(result)
    assert qt.is_cuda
    assert qt.shape == (2, 4)
    assert qt.dtype == torch.complex64

    expected = torch.tensor(
        [
            [0.6, 0.8, 0.0, 0.0],
            [
                1.0 / (10.0**0.5),
                2.0 / (10.0**0.5),
                2.0 / (10.0**0.5),
                1.0 / (10.0**0.5),
            ],
        ],
        dtype=torch.complex64,
        device="cuda",
    )
    assert torch.allclose(qt, expected)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_float32_amplitude_2d_respects_engine_precision() -> None:
    """2D float32 CUDA amplitude batch should still honor float64 engine output precision."""
    engine = QdpEngine(0, precision="float64")
    t = torch.tensor(
        [[3.0, 4.0, 0.0, 0.0], [1.0, 2.0, 2.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )

    result = engine.encode(t, num_qubits=2, encoding_method="amplitude")
    assert result is not None

    qt = torch.from_dlpack(result)
    assert qt.is_cuda
    assert qt.shape == (2, 4)
    assert qt.dtype == torch.complex128

    expected = torch.tensor(
        [
            [0.6, 0.8, 0.0, 0.0],
            [
                1.0 / (10.0**0.5),
                2.0 / (10.0**0.5),
                2.0 / (10.0**0.5),
                1.0 / (10.0**0.5),
            ],
        ],
        dtype=torch.complex128,
        device="cuda",
    )
    assert torch.allclose(qt, expected)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_float32_angle_supported_single_sample() -> None:
    """1D float32 CUDA tensor should be supported for single-sample angle encoding."""
    engine = _engine()
    t = torch.tensor([torch.pi / 2, 0.0], dtype=torch.float32, device="cuda")

    result = engine.encode(t, num_qubits=2, encoding_method="angle")
    assert result is not None

    qt = torch.from_dlpack(result)
    assert qt.is_cuda
    assert qt.shape == (1, 4)
    assert qt.dtype == torch.complex64

    expected = torch.tensor(
        [[0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j]],
        dtype=torch.complex64,
        device="cuda",
    )
    assert torch.allclose(qt, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_float32_angle_2d_supported() -> None:
    """2D float32 CUDA tensor should use the batch GPU-pointer float32 angle path."""
    engine = _engine()
    t = torch.tensor(
        [[0.0, 0.0], [torch.pi / 2, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )

    result = engine.encode(t, num_qubits=2, encoding_method="angle")
    assert result is not None

    qt = torch.from_dlpack(result)
    assert qt.is_cuda
    assert qt.shape == (2, 4)
    assert qt.dtype == torch.complex64

    expected = torch.tensor(
        [
            [1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j],
        ],
        dtype=torch.complex64,
        device="cuda",
    )
    assert torch.allclose(qt, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_float32_angle_non_contiguous_rejected() -> None:
    """1D float32 CUDA angle tensor must still be contiguous."""
    engine = _engine()
    t = torch.randn(4, dtype=torch.float32, device="cuda")[::2]
    assert t.stride(0) != 1

    with pytest.raises(RuntimeError, match="contiguous"):
        engine.encode(t, num_qubits=2, encoding_method="angle")


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_float16_angle_rejected() -> None:
    """Angle encoding should not accept float16 CUDA tensors."""
    engine = _engine()
    t = torch.tensor([0.0, torch.pi / 2], dtype=torch.float16, device="cuda")

    with pytest.raises(RuntimeError, match="float64 or float32 for angle encoding"):
        engine.encode(t, num_qubits=2, encoding_method="angle")


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_stride_1d_non_contiguous_rejected() -> None:
    """Non-contiguous 1D CUDA tensor (stride != 1) should fail with contiguous requirement."""
    engine = _engine()
    # Slice so stride is 2: shape (2,), stride (2,)
    t = torch.randn(4, dtype=torch.float64, device="cuda")[::2]
    assert t.stride(0) != 1
    with pytest.raises(RuntimeError) as exc_info:
        engine.encode(t, num_qubits=1, encoding_method="amplitude")
    msg = str(exc_info.value).lower()
    assert "contiguous" in msg
    # Accept either explicit stride[0]/expected or user-facing contiguous() hint
    assert "stride" in msg or "contiguous()" in msg or "expected" in msg


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_stride_2d_non_contiguous_rejected() -> None:
    """Non-contiguous 2D CUDA tensor should fail with contiguous requirement."""
    engine = _engine()
    # (4, 2) with strides (3, 2) -> not C-contiguous; expected for (4,2) is (2, 1)
    t = torch.randn(4, 3, dtype=torch.float64, device="cuda")[:, ::2]
    assert t.dim() == 2
    assert t.shape == (4, 2)
    # Strides should be (3, 2) not (2, 1)
    assert t.stride(0) == 3
    assert t.stride(1) == 2
    with pytest.raises(RuntimeError) as exc_info:
        engine.encode(t, num_qubits=1, encoding_method="amplitude")
    msg = str(exc_info.value).lower()
    assert "contiguous" in msg
    # Accept either explicit strides=/expected or user-facing contiguous() hint
    assert "stride" in msg or "contiguous()" in msg or "expected" in msg


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_valid_cuda_float64_1d_succeeds() -> None:
    """Valid 1D float64 contiguous CUDA tensor should encode successfully."""
    engine = _engine()
    t = torch.randn(4, dtype=torch.float64, device="cuda")
    result = engine.encode(t, num_qubits=2, encoding_method="amplitude")
    assert result is not None


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_valid_cuda_float64_2d_succeeds() -> None:
    """Valid 2D float64 contiguous CUDA tensor should encode successfully."""
    engine = _engine()
    t = torch.randn(3, 4, dtype=torch.float64, device="cuda")
    result = engine.encode(t, num_qubits=2, encoding_method="amplitude")
    assert result is not None
