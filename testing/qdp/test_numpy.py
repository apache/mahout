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

import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from .qdp_test_utils import requires_qdp


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")


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
    ("num_samples", "num_qubits", "check_norm"),
    [
        (10, 3, True),  # Basic: 10 samples, 3 qubits, check normalization
        (100, 6, False),  # Large: 100 samples, 6 qubits
        (1, 4, False),  # Single sample: 1 sample, 4 qubits
    ],
)
def test_encode_from_numpy_file(num_samples, num_qubits, check_norm):
    """Test NumPy file encoding"""
    from _qdp import QdpEngine

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

    engine = QdpEngine(device_id=0)
    sample_size = 2**num_qubits
    data = np.random.randn(sample_size).astype(np.float64)
    data = data / np.linalg.norm(data)

    qtensor = engine.encode(data, num_qubits)
    tensor = torch.from_dlpack(qtensor)
    _verify_tensor(tensor, (1, sample_size), check_normalization=True)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(("num_samples", "num_qubits"), [(5, 2), (10, 3)])
def test_encode_numpy_array_2d(num_samples, num_qubits):
    """Test 2D NumPy array encoding (batch)"""
    from _qdp import QdpEngine

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
    """Test amplitude encoding via encoding_method parameter"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    num_qubits = 2
    sample_size = 2**num_qubits
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    qtensor = engine.encode(data, num_qubits, encoding_method=encoding_method)
    tensor = torch.from_dlpack(qtensor)
    _verify_tensor(tensor, (1, sample_size))


# ---------------------------------------------------------------------------
# Angle encoding tests
# ---------------------------------------------------------------------------


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_encode_numpy_angle_encoding_1d(num_qubits):
    """Angle encoding: 1D array with num_qubits angles (single sample)"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    # Angle encoding expects exactly num_qubits values: one angle per qubit
    angles = np.random.uniform(0, 2 * np.pi, size=num_qubits).astype(np.float64)

    qtensor = engine.encode(angles, num_qubits, encoding_method="angle")
    tensor = torch.from_dlpack(qtensor)

    _verify_tensor(tensor, (1, 2**num_qubits), check_normalization=True)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(("num_samples", "num_qubits"), [(5, 2), (10, 3), (1, 4)])
def test_encode_numpy_angle_encoding_2d(num_samples, num_qubits):
    """Angle encoding: 2D array of shape (num_samples, num_qubits)"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    angles = np.random.uniform(0, 2 * np.pi, size=(num_samples, num_qubits)).astype(
        np.float64
    )

    qtensor = engine.encode(angles, num_qubits, encoding_method="angle")
    tensor = torch.from_dlpack(qtensor)

    _verify_tensor(tensor, (num_samples, 2**num_qubits), check_normalization=True)


@requires_qdp
@pytest.mark.gpu
def test_encode_numpy_angle_encoding_from_file():
    """Angle encoding: load angles from .npy file"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    num_qubits = 3
    num_samples = 8
    angles = np.random.uniform(0, 2 * np.pi, size=(num_samples, num_qubits)).astype(
        np.float64
    )

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_path = f.name
    try:
        np.save(npy_path, angles)
        qtensor = engine.encode(npy_path, num_qubits, encoding_method="angle")
        tensor = torch.from_dlpack(qtensor)
        _verify_tensor(tensor, (num_samples, 2**num_qubits), check_normalization=True)
    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)


@requires_qdp
@pytest.mark.gpu
def test_encode_numpy_angle_encoding_wrong_sample_size():
    """Angle encoding: wrong sample_size raises an error"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    num_qubits = 3
    # Pass 2^num_qubits values instead of num_qubits values
    wrong_data = np.ones(2**num_qubits, dtype=np.float64)

    with pytest.raises((RuntimeError, ValueError)):
        engine.encode(wrong_data, num_qubits, encoding_method="angle")


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("theta", "expected_probs"),
    [
        pytest.param(0.0, [1.0, 0.0], id="theta=0"),
        pytest.param(np.pi / 2, [0.5, 0.5], id="theta=pi/2"),
        pytest.param(np.pi, [0.0, 1.0], id="theta=pi"),
    ],
)
def test_encode_numpy_angle_encoding_1qubit_correctness(theta, expected_probs):
    """Angle encoding: 1 qubit with known angle θ → probabilities [cos²(θ/2), sin²(θ/2)]"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0, precision="float64")
    data = np.array([theta], dtype=np.float64)

    qtensor = engine.encode(data, 1, encoding_method="angle")
    tensor = torch.from_dlpack(qtensor)

    probs = tensor.abs().pow(2).squeeze(0)  # shape: (2,)
    expected = torch.tensor(expected_probs, dtype=probs.dtype, device=probs.device)

    assert torch.allclose(probs, expected, atol=1e-6), (
        f"For θ={theta}, expected {expected.cpu().tolist()}, got {probs.cpu().tolist()}"
    )


# ---------------------------------------------------------------------------
# Basis encoding tests
# ---------------------------------------------------------------------------


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_encode_numpy_basis_encoding_1d(num_qubits):
    """Basis encoding: 1D array with a single index (single sample)"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    # Basis encoding expects sample_size=1: one integer index per sample
    index = np.array([float(2**num_qubits - 1)], dtype=np.float64)

    qtensor = engine.encode(index, num_qubits, encoding_method="basis")
    tensor = torch.from_dlpack(qtensor)

    _verify_tensor(tensor, (1, 2**num_qubits), check_normalization=True)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(("num_samples", "num_qubits"), [(5, 2), (10, 3), (1, 4)])
def test_encode_numpy_basis_encoding_2d(num_samples, num_qubits):
    """Basis encoding: 2D array of shape (num_samples, 1) with integer indices"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    max_index = 2**num_qubits
    indices = np.random.randint(0, max_index, size=(num_samples, 1)).astype(np.float64)

    qtensor = engine.encode(indices, num_qubits, encoding_method="basis")
    tensor = torch.from_dlpack(qtensor)

    _verify_tensor(tensor, (num_samples, 2**num_qubits), check_normalization=True)


@requires_qdp
@pytest.mark.gpu
def test_encode_numpy_basis_encoding_from_file():
    """Basis encoding: load indices from .npy file"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    num_qubits = 3
    num_samples = 6
    indices = np.random.randint(0, 2**num_qubits, size=(num_samples, 1)).astype(
        np.float64
    )

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_path = f.name
    try:
        np.save(npy_path, indices)
        qtensor = engine.encode(npy_path, num_qubits, encoding_method="basis")
        tensor = torch.from_dlpack(qtensor)
        _verify_tensor(tensor, (num_samples, 2**num_qubits), check_normalization=True)
    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("basis_index", "num_qubits"),
    [
        (0, 2),  # |00⟩: amplitude 1 at index 0
        (1, 2),  # |01⟩: amplitude 1 at index 1
        (3, 2),  # |11⟩: amplitude 1 at index 3
        (0, 3),  # |000⟩: amplitude 1 at index 0
        (13, 4),  # |1101⟩: amplitude 1 at index 13
    ],
)
def test_encode_numpy_basis_state_correctness(basis_index, num_qubits):
    """Basis encoding: verify the encoded state is exactly |basis_index⟩"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0, precision="float64")
    data = np.array([[float(basis_index)]], dtype=np.float64)

    qtensor = engine.encode(data, num_qubits, encoding_method="basis")
    tensor = torch.from_dlpack(qtensor)

    # The encoded state should be a one-hot complex vector with |amplitude|²=1
    # at basis_index and 0 elsewhere
    probs = tensor.abs().pow(2).squeeze(0)  # shape: (2^num_qubits,)
    expected = torch.zeros(2**num_qubits, dtype=probs.dtype, device=probs.device)
    expected[basis_index] = 1.0

    assert torch.allclose(probs, expected, atol=1e-6), (
        f"Expected |{basis_index}⟩ but got probabilities {probs.cpu().tolist()}"
    )


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bad_data",
    [
        pytest.param(np.array([[-1.0]], dtype=np.float64), id="negative index"),
        pytest.param(np.array([[0.5]], dtype=np.float64), id="non-integer index"),
    ],
)
def test_encode_numpy_basis_encoding_invalid_index(bad_data):
    """Basis encoding: invalid index values raise an error"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    with pytest.raises((RuntimeError, ValueError)):
        engine.encode(bad_data, 2, encoding_method="basis")


@requires_qdp
@pytest.mark.gpu
def test_encode_numpy_basis_encoding_out_of_range():
    """Basis encoding: index >= 2^num_qubits raises an error"""
    from _qdp import QdpEngine

    engine = QdpEngine(device_id=0)
    num_qubits = 2
    out_of_range = np.array([[float(2**num_qubits)]], dtype=np.float64)  # index == 4

    with pytest.raises((RuntimeError, ValueError)):
        engine.encode(out_of_range, num_qubits, encoding_method="basis")


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("float32", torch.complex64),
        ("float64", torch.complex128),
    ],
)
def test_encode_numpy_precision(precision, expected_dtype):
    """Test different precision settings"""
    from _qdp import QdpEngine

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
    ("data", "error_match"),
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

    engine = QdpEngine(device_id=0)
    num_qubits = 2 if data.ndim == 1 else 1

    with pytest.raises((RuntimeError, TypeError)):
        engine.encode(data, num_qubits)
