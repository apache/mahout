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

try:
    import torch
    from _qdp import QdpEngine
except ImportError:
    torch = None
    QdpEngine = None


def _cuda_available():
    return torch is not None and torch.cuda.is_available()


def _engine():
    if QdpEngine is None:
        pytest.skip("_qdp not built")
    e = QdpEngine(0)
    return e


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_dtype_validation_float32_rejected():
    """DLPack tensor must be float64; float32 CUDA tensor should fail with clear message."""
    engine = _engine()
    # 1D float32 CUDA tensor (contiguous)
    t = torch.randn(4, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError) as exc_info:
        engine.encode(t, num_qubits=2, encoding_method="amplitude")
    msg = str(exc_info.value).lower()
    assert "float64" in msg
    # Accept either DLPack-style (code=/bits=/lanes=) or user-facing (float32/dtype) message
    assert (
        "code=" in msg
        or "bits=" in msg
        or "lanes=" in msg
        or "float32" in msg
        or "dtype" in msg
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_stride_1d_non_contiguous_rejected():
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
def test_stride_2d_non_contiguous_rejected():
    """Non-contiguous 2D CUDA tensor should fail with contiguous requirement."""
    engine = _engine()
    # (4, 2) with strides (3, 2) -> not C-contiguous; expected for (4,2) is (2, 1)
    t = torch.randn(4, 3, dtype=torch.float64, device="cuda")[:, ::2]
    assert t.dim() == 2 and t.shape == (4, 2)
    # Strides should be (3, 2) not (2, 1)
    assert t.stride(0) == 3 and t.stride(1) == 2
    with pytest.raises(RuntimeError) as exc_info:
        engine.encode(t, num_qubits=1, encoding_method="amplitude")
    msg = str(exc_info.value).lower()
    assert "contiguous" in msg
    # Accept either explicit strides=/expected or user-facing contiguous() hint
    assert "stride" in msg or "contiguous()" in msg or "expected" in msg


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_valid_cuda_float64_1d_succeeds():
    """Valid 1D float64 contiguous CUDA tensor should encode successfully."""
    engine = _engine()
    t = torch.randn(4, dtype=torch.float64, device="cuda")
    result = engine.encode(t, num_qubits=2, encoding_method="amplitude")
    assert result is not None


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_valid_cuda_float64_2d_succeeds():
    """Valid 2D float64 contiguous CUDA tensor should encode successfully."""
    engine = _engine()
    t = torch.randn(3, 4, dtype=torch.float64, device="cuda")
    result = engine.encode(t, num_qubits=2, encoding_method="amplitude")
    assert result is not None
