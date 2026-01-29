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

"""Test NumPy file format and array input support in Mahout QDP Python bindings"""

import tempfile
import os

import numpy as np
import pytest
import torch

from .qdp_test_utils import requires_qdp


def _verify_tensor(tensor, expected_shape, check_normalization=False):
    """Helper function to verify tensor properties"""
    assert tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {tensor.shape}"
    )
    assert tensor.is_cuda, "Tensor should be on CUDA device"

    if check_normalization:
        norms = tensor.abs().pow(2).sum(dim=1).sqrt()
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            "States should be normalized"
        )


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "num_samples,num_qubits,check_norm",
    [
        (10, 3, True),  # Basic: 10 samples, 3 qubits, check normalization
        (100, 6, False),  # Large: 100 samples, 6 qubits
        (1, 4, False),  # Single sample: 1 sample, 4 qubits
    ],
)
def test_encode_from_numpy_file(num_samples, num_qubits, check_norm):
    """Test NumPy file encoding"""
    from _qdp import QdpEngine

    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(device_id=0)
    sample_size = 2**num_qubits

    # Generate normalized data
    data = np.random.randn(num_samples, sample_size).astype(np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms

    # Save to temporary .npy file
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_path = f.name

    try:
        np.save(npy_path, data)
        qtensor = engine.encode(npy_path, num_qubits)
        tensor = torch.from_dlpack(qtensor)

        _verify_tensor(tensor, (num_samples, sample_size), check_norm)

    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_encode_numpy_array_1d(num_qubits):
    """Test 1D NumPy array encoding (single sample)"""
    from _qdp import QdpEngine

    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(device_id=0)
    sample_size = 2**num_qubits
    data = np.random.randn(sample_size).astype(np.float64)
    data = data / np.linalg.norm(data)

    qtensor = engine.encode(data, num_qubits)
    tensor = torch.from_dlpack(qtensor)
    _verify_tensor(tensor, (1, sample_size), check_normalization=True)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("num_samples,num_qubits", [(5, 2), (10, 3)])
def test_encode_numpy_array_2d(num_samples, num_qubits):
    """Test 2D NumPy array encoding (batch)"""
    from _qdp import QdpEngine

    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(device_id=0)
    sample_size = 2**num_qubits
    data = np.random.randn(num_samples, sample_size).astype(np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms

    qtensor = engine.encode(data, num_qubits)
    tensor = torch.from_dlpack(qtensor)
    _verify_tensor(tensor, (num_samples, sample_size), check_normalization=True)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("encoding_method", ["amplitude"])
def test_encode_numpy_encoding_methods(encoding_method):
    """Test different encoding methods"""
    from _qdp import QdpEngine

    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    # TODO: Add angle and basis encoding tests when implemented
    engine = QdpEngine(device_id=0)
    num_qubits = 2
    sample_size = 2**num_qubits
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    qtensor = engine.encode(data, num_qubits, encoding_method=encoding_method)
    tensor = torch.from_dlpack(qtensor)
    _verify_tensor(tensor, (1, sample_size))


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "precision,expected_dtype",
    [
        ("float32", torch.complex64),
        ("float64", torch.complex128),
    ],
)
def test_encode_numpy_precision(precision, expected_dtype):
    """Test different precision settings"""
    from _qdp import QdpEngine

    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(device_id=0, precision=precision)
    num_qubits = 2
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    qtensor = engine.encode(data, num_qubits)
    tensor = torch.from_dlpack(qtensor)
    assert tensor.dtype == expected_dtype, (
        f"Expected {expected_dtype}, got {tensor.dtype}"
    )


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "data,error_match",
    [
        (
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            None,  # Wrong dtype - will raise RuntimeError or TypeError
        ),
        (
            np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64),
            None,  # 3D array - will raise RuntimeError or TypeError
        ),
    ],
)
def test_encode_numpy_errors(data, error_match):
    """Test error handling for invalid inputs"""
    from _qdp import QdpEngine

    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(device_id=0)
    num_qubits = 2 if data.ndim == 1 else 1

    with pytest.raises((RuntimeError, TypeError)):
        engine.encode(data, num_qubits)
