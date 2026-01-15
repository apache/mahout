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
import torch
from _qdp import QdpEngine


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


def test_encode_from_numpy_file():
    """Test NumPy file encoding"""
    engine = QdpEngine(device_id=0)

    test_cases = [
        (10, 3, True),  # Basic: 10 samples, 3 qubits, check normalization
        (100, 6, False),  # Large: 100 samples, 6 qubits
        (1, 4, False),  # Single sample: 1 sample, 4 qubits
    ]

    for num_samples, num_qubits, check_norm in test_cases:
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


def test_encode_numpy_array():
    """Test NumPy array encoding (1D and 2D)"""
    engine = QdpEngine(device_id=0)

    # 1D arrays
    for num_qubits in [1, 2, 3, 4]:
        sample_size = 2**num_qubits
        data = np.random.randn(sample_size).astype(np.float64)
        data = data / np.linalg.norm(data)

        qtensor = engine.encode(data, num_qubits)
        tensor = torch.from_dlpack(qtensor)
        _verify_tensor(tensor, (1, sample_size), check_normalization=True)

    # 2D arrays
    for num_samples, num_qubits in [(5, 2), (10, 3)]:
        sample_size = 2**num_qubits
        data = np.random.randn(num_samples, sample_size).astype(np.float64)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = data / norms

        qtensor = engine.encode(data, num_qubits)
        tensor = torch.from_dlpack(qtensor)
        _verify_tensor(tensor, (num_samples, sample_size), check_normalization=True)


def test_encode_numpy_configurations():
    """Test different encoding methods and precision settings"""
    num_qubits = 2
    sample_size = 2**num_qubits
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    # Test encoding methods
    # TODO: Add angle and basis encoding tests when implemented
    engine = QdpEngine(device_id=0)
    for method in ["amplitude"]:
        qtensor = engine.encode(data, num_qubits, encoding_method=method)
        tensor = torch.from_dlpack(qtensor)
        _verify_tensor(tensor, (1, sample_size))

    # Test precision settings
    for precision, expected_dtype in [
        ("float32", torch.complex64),
        ("float64", torch.complex128),
    ]:
        engine = QdpEngine(device_id=0, precision=precision)
        qtensor = engine.encode(data, num_qubits)
        tensor = torch.from_dlpack(qtensor)
        assert tensor.dtype == expected_dtype, (
            f"Expected {expected_dtype}, got {tensor.dtype}"
        )


def test_encode_numpy_errors():
    """Test error handling for invalid inputs"""
    engine = QdpEngine(device_id=0)

    # Test wrong dtype
    data_wrong_dtype = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        engine.encode(data_wrong_dtype, 2)
        assert False, "Should have raised an error for wrong dtype"
    except (RuntimeError, TypeError):
        pass  # Expected

    # Test unsupported dimensions (3D+)
    data_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
    try:
        engine.encode(data_3d, 1)
        assert False, "Should have raised an error for 3D array"
    except (RuntimeError, TypeError):
        pass  # Expected
