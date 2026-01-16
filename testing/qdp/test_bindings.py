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


def _has_multi_gpu():
    """Check if multiple GPUs are available via PyTorch."""
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except ImportError:
        return False


def test_import():
    """Test that PyO3 bindings are properly imported."""
    assert hasattr(_qdp, "QdpEngine")
    assert hasattr(_qdp, "QuantumTensor")

    # Test that QdpEngine has the new encode_from_tensorflow method
    from _qdp import QdpEngine

    assert hasattr(QdpEngine, "encode_from_tensorflow")
    assert callable(getattr(QdpEngine, "encode_from_tensorflow"))


@pytest.mark.gpu
def test_encode():
    """Test encoding returns QuantumTensor (requires GPU)."""
    from _qdp import QdpEngine

    engine = QdpEngine(0)
    data = [0.5, 0.5, 0.5, 0.5]
    qtensor = engine.encode(data, 2, "amplitude")
    assert isinstance(qtensor, _qdp.QuantumTensor)


@pytest.mark.gpu
def test_dlpack_device():
    """Test __dlpack_device__ method (requires GPU)."""
    from _qdp import QdpEngine

    engine = QdpEngine(0)
    data = [1.0, 2.0, 3.0, 4.0]
    qtensor = engine.encode(data, 2, "amplitude")

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

    # Test with device_id=1 (second GPU)
    device_id = 1
    engine = QdpEngine(device_id)
    data = [1.0, 2.0, 3.0, 4.0]
    qtensor = engine.encode(data, 2, "amplitude")

    device_info = qtensor.__dlpack_device__()
    assert device_info == (
        2,
        device_id,
    ), f"Expected (2, {device_id}) for CUDA device {device_id}"

    # Verify PyTorch integration works with non-zero device_id
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.is_cuda
    assert torch_tensor.device.index == device_id, (
        f"PyTorch tensor should be on device {device_id}"
    )


@pytest.mark.gpu
def test_dlpack_single_use():
    """Test that __dlpack__ can only be called once (requires GPU)."""
    import torch
    from _qdp import QdpEngine

    engine = QdpEngine(0)
    data = [1.0, 2.0, 3.0, 4.0]
    qtensor = engine.encode(data, 2, "amplitude")

    # First call succeeds - let PyTorch consume it
    _ = torch.from_dlpack(qtensor)

    # Second call should fail because tensor was already consumed
    qtensor2 = engine.encode(data, 2, "amplitude")
    _ = qtensor2.__dlpack__()  # Consume the capsule
    with pytest.raises(RuntimeError, match="already consumed"):
        qtensor2.__dlpack__()


@pytest.mark.gpu
def test_pytorch_integration():
    """Test PyTorch integration via DLPack (requires GPU and PyTorch)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    engine = QdpEngine(0)
    data = [1.0, 2.0, 3.0, 4.0]
    qtensor = engine.encode(data, 2, "amplitude")

    # Convert to PyTorch tensor using DLPack
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.is_cuda
    assert torch_tensor.device.index == 0
    assert torch_tensor.dtype == torch.complex64

    # Verify shape (2 qubits = 2^2 = 4 elements) as 2D for consistency: [1, 4]
    assert torch_tensor.shape == (1, 4)


@pytest.mark.gpu
def test_pytorch_precision_float64():
    """Verify optional float64 precision produces complex128 tensors."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    engine = QdpEngine(0, precision="float64")
    data = [1.0, 2.0, 3.0, 4.0]
    qtensor = engine.encode(data, 2, "amplitude")

    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.dtype == torch.complex128


@pytest.mark.gpu
def test_encode_tensor_cpu():
    """Test encoding from CPU PyTorch tensor (1D, single sample)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    qtensor = engine.encode(data, 2, "amplitude")

    # Verify result
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.is_cuda
    assert torch_tensor.shape == (1, 4)


@pytest.mark.gpu
def test_encode_tensor_batch():
    """Test encoding from CPU PyTorch tensor (2D, batch encoding with zero-copy)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)
    # Create 2D tensor (batch_size=3, features=4)
    data = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=torch.float64,
    )
    assert data.is_contiguous(), "Test tensor should be contiguous for zero-copy"

    qtensor = engine.encode(data, 2, "amplitude")

    # Verify result
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.is_cuda
    assert torch_tensor.shape == (3, 4), "Batch encoding should preserve batch size"


@pytest.mark.gpu
def test_encode_errors():
    """Test error handling for unified encode method."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Test unsupported file format
    with pytest.raises(RuntimeError, match="Unsupported file format"):
        engine.encode("data.txt", 2, "amplitude")

    # Test unsupported data type
    with pytest.raises(RuntimeError, match="Unsupported data type"):
        engine.encode({"key": "value"}, 2, "amplitude")

    # Test GPU tensor input (should fail as only CPU is supported)
    gpu_tensor = torch.tensor([1.0, 2.0], device="cuda:0")
    with pytest.raises(RuntimeError, match="Only CPU tensors are currently supported"):
        engine.encode(gpu_tensor, 1, "amplitude")
