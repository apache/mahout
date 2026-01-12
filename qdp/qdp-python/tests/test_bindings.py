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

"""Simple tests for PyO3 bindings."""

import pytest
import _qdp

TEST_DATA_1D = [1.0, 2.0, 3.0, 4.0]
TEST_DATA_1D_NORMALIZED = [0.5, 0.5, 0.5, 0.5]
NUM_QUBITS = 2
SAMPLE_SIZE = 4  # 2^NUM_QUBITS


def _has_multi_gpu():
    """Check if multiple GPUs are available via PyTorch."""
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except ImportError:
        return False


@pytest.fixture
def engine():
    """Create QdpEngine instance for testing."""
    from _qdp import QdpEngine

    return QdpEngine(0)


@pytest.fixture
def engine_float64():
    """Create QdpEngine instance with float64 precision."""
    from _qdp import QdpEngine

    return QdpEngine(0, precision="float64")


def test_import():
    """Test that PyO3 bindings are properly imported."""
    assert hasattr(_qdp, "QdpEngine")
    assert hasattr(_qdp, "QuantumTensor")

    # Test that QdpEngine has the new encode_from_tensorflow method
    from _qdp import QdpEngine

    assert hasattr(QdpEngine, "encode_from_tensorflow")
    assert callable(getattr(QdpEngine, "encode_from_tensorflow"))


@pytest.mark.gpu
def test_encode(engine):
    """Test encoding returns QuantumTensor (requires GPU)."""
    qtensor = engine.encode(TEST_DATA_1D_NORMALIZED, NUM_QUBITS, "amplitude")
    assert isinstance(qtensor, _qdp.QuantumTensor)


@pytest.mark.gpu
def test_dlpack_device(engine):
    """Test __dlpack_device__ method (requires GPU)."""
    qtensor = engine.encode(TEST_DATA_1D, NUM_QUBITS, "amplitude")
    device_info = qtensor.__dlpack_device__()
    assert device_info == (2, 0), "Expected (2, 0) for CUDA device 0"


@pytest.mark.gpu
@pytest.mark.skipif(
    not _has_multi_gpu(), reason="Multi-GPU setup required for this test"
)
def test_dlpack_device_id_non_zero():
    """Test device_id propagation for non-zero devices (requires multi-GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    device_id = 1
    engine = QdpEngine(device_id)
    qtensor = engine.encode(TEST_DATA_1D, NUM_QUBITS, "amplitude")

    device_info = qtensor.__dlpack_device__()
    assert device_info == (2, device_id), (
        f"Expected (2, {device_id}) for CUDA device {device_id}"
    )

    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.is_cuda
    assert torch_tensor.device.index == device_id


@pytest.mark.gpu
def test_dlpack_single_use(engine):
    """Test that __dlpack__ can only be called once (requires GPU)."""
    import torch

    qtensor = engine.encode(TEST_DATA_1D, NUM_QUBITS, "amplitude")
    _ = torch.from_dlpack(qtensor)

    qtensor2 = engine.encode(TEST_DATA_1D, NUM_QUBITS, "amplitude")
    _ = qtensor2.__dlpack__()
    with pytest.raises(RuntimeError, match="already consumed"):
        qtensor2.__dlpack__()


@pytest.mark.gpu
def test_pytorch_integration(engine):
    """Test PyTorch integration via DLPack (requires GPU and PyTorch)."""
    pytest.importorskip("torch")
    import torch

    qtensor = engine.encode(TEST_DATA_1D, NUM_QUBITS, "amplitude")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.is_cuda
    assert torch_tensor.device.index == 0
    assert torch_tensor.dtype == torch.complex64
    assert torch_tensor.shape == (1, SAMPLE_SIZE)


@pytest.mark.gpu
def test_pytorch_precision_float64(engine_float64):
    """Verify optional float64 precision produces complex128 tensors."""
    pytest.importorskip("torch")
    import torch

    qtensor = engine_float64.encode(TEST_DATA_1D, NUM_QUBITS, "amplitude")
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.dtype == torch.complex128


@pytest.mark.gpu
def test_encode_pytorch_tensor(engine):
    """Test encoding from CPU PyTorch tensor (1D and 2D)."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    # Test 1D tensor
    data_1d = torch.tensor(TEST_DATA_1D, dtype=torch.float64)
    qtensor_1d = engine.encode(data_1d, NUM_QUBITS, "amplitude")
    torch_tensor_1d = torch.from_dlpack(qtensor_1d)
    assert torch_tensor_1d.is_cuda and torch_tensor_1d.shape == (1, SAMPLE_SIZE)

    # Test 2D tensor
    data_2d = torch.randn(3, SAMPLE_SIZE, dtype=torch.float64)
    qtensor_2d = engine.encode(data_2d, NUM_QUBITS, "amplitude")
    torch_tensor_2d = torch.from_dlpack(qtensor_2d)
    assert torch_tensor_2d.is_cuda and torch_tensor_2d.shape == (3, SAMPLE_SIZE)


@pytest.mark.gpu
def test_encode_errors(engine):
    """Test error handling for unified encode method."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    with pytest.raises(RuntimeError, match="Unsupported file format"):
        engine.encode("data.txt", NUM_QUBITS, "amplitude")

    with pytest.raises(RuntimeError, match="Unsupported data type"):
        engine.encode({"key": "value"}, NUM_QUBITS, "amplitude")

    gpu_tensor = torch.tensor([1.0, 2.0], device="cuda:0")
    with pytest.raises(RuntimeError, match="Only CPU tensors are currently supported"):
        engine.encode(gpu_tensor, 1, "amplitude")


@pytest.mark.gpu
def test_encode_numpy_array(engine):
    """Test encoding from NumPy array (1D and 2D)."""
    import torch
    import numpy as np

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    # Test 1D array
    data_1d = np.array(TEST_DATA_1D, dtype=np.float64)
    qtensor_1d = engine.encode(
        data_1d, num_qubits=NUM_QUBITS, encoding_method="amplitude"
    )
    torch_tensor_1d = torch.from_dlpack(qtensor_1d)
    assert torch_tensor_1d.is_cuda and torch_tensor_1d.shape == (1, SAMPLE_SIZE)

    # Test 2D array
    data_2d = np.random.randn(5, SAMPLE_SIZE).astype(np.float64)
    qtensor_2d = engine.encode(
        data_2d, num_qubits=NUM_QUBITS, encoding_method="amplitude"
    )
    torch_tensor_2d = torch.from_dlpack(qtensor_2d)
    assert torch_tensor_2d.is_cuda and torch_tensor_2d.shape == (5, SAMPLE_SIZE)


@pytest.mark.gpu
def test_encode_pathlib_path(engine):
    """Test encoding from pathlib.Path object."""
    from pathlib import Path
    import tempfile
    import os
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    data = np.random.randn(2, SAMPLE_SIZE).astype(np.float64)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_path = f.name

    try:
        np.save(npy_path, data)
        qtensor = engine.encode(
            Path(npy_path), num_qubits=NUM_QUBITS, encoding_method="amplitude"
        )
        torch_tensor = torch.from_dlpack(qtensor)
        assert torch_tensor.is_cuda and torch_tensor.shape == (2, SAMPLE_SIZE)
    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)
