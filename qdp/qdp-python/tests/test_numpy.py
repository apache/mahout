#!/usr/bin/env python3
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

"""Test NumPy file format support in Mahout QDP Python bindings"""

import tempfile
import os
import numpy as np
import torch
from _qdp import QdpEngine


def test_encode_from_numpy_basic():
    """Test basic NumPy file encoding"""
    engine = QdpEngine(device_id=0)

    # Create test data
    num_samples = 10
    num_qubits = 3
    sample_size = 2**num_qubits  # 8

    # Generate normalized data
    data = np.random.randn(num_samples, sample_size).astype(np.float64)
    # Normalize each row
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms

    # Save to temporary .npy file
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_path = f.name

    try:
        np.save(npy_path, data)

        # Encode from NumPy file
        qtensor = engine.encode_from_numpy(npy_path, num_qubits, "amplitude")

        # Convert to PyTorch
        tensor = torch.from_dlpack(qtensor)

        # Verify shape
        assert tensor.shape == (num_samples, sample_size), (
            f"Expected shape {(num_samples, sample_size)}, got {tensor.shape}"
        )

        # Verify it's on GPU
        assert tensor.is_cuda, "Tensor should be on CUDA device"

        # Verify normalization (amplitude encoding normalizes)
        norms = tensor.abs().pow(2).sum(dim=1).sqrt()
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            "States should be normalized"
        )

        print("✓ test_encode_from_numpy_basic passed")

    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)


def test_encode_from_numpy_large():
    """Test NumPy encoding with larger dataset"""
    engine = QdpEngine(device_id=0)

    num_samples = 100
    num_qubits = 6
    sample_size = 2**num_qubits  # 64

    # Generate test data
    data = np.random.randn(num_samples, sample_size).astype(np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms

    # Save to temporary .npy file
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_path = f.name

    try:
        np.save(npy_path, data)

        # Encode
        qtensor = engine.encode_from_numpy(npy_path, num_qubits, "amplitude")
        tensor = torch.from_dlpack(qtensor)

        # Verify
        assert tensor.shape == (num_samples, sample_size)
        assert tensor.is_cuda

        print("✓ test_encode_from_numpy_large passed")

    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)


def test_encode_from_numpy_single_sample():
    """Test NumPy encoding with single sample"""
    engine = QdpEngine(device_id=0)

    num_qubits = 4
    sample_size = 2**num_qubits  # 16

    # Single sample
    data = np.random.randn(1, sample_size).astype(np.float64)
    data = data / np.linalg.norm(data)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_path = f.name

    try:
        np.save(npy_path, data)

        qtensor = engine.encode_from_numpy(npy_path, num_qubits, "amplitude")
        tensor = torch.from_dlpack(qtensor)

        assert tensor.shape == (1, sample_size)
        assert tensor.is_cuda

        print("✓ test_encode_from_numpy_single_sample passed")

    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)


if __name__ == "__main__":
    test_encode_from_numpy_basic()
    test_encode_from_numpy_large()
    test_encode_from_numpy_single_sample()
    print("\n✅ All NumPy encoding tests passed!")
