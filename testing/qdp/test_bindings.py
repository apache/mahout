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
def test_encode_from_tensorflow_binding():
    """Test TensorFlow TensorProto binding path (requires GPU and TensorFlow)."""
    pytest.importorskip("torch")
    tf = pytest.importorskip("tensorflow")
    import numpy as np
    import torch
    from _qdp import QdpEngine
    import os
    import tempfile

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)
    num_qubits = 2
    sample_size = 2**num_qubits

    data = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float64)
    tensor_proto = tf.make_tensor_proto(data, dtype=tf.float64)

    with tempfile.NamedTemporaryFile(suffix=".pb", delete=False) as f:
        pb_path = f.name
        f.write(tensor_proto.SerializeToString())

    try:
        qtensor = engine.encode_from_tensorflow(pb_path, num_qubits, "amplitude")
        torch_tensor = torch.from_dlpack(qtensor)
        assert torch_tensor.is_cuda
        assert torch_tensor.shape == (2, sample_size)
    finally:
        if os.path.exists(pb_path):
            os.remove(pb_path)


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


@pytest.mark.gpu
def test_basis_encode_basic():
    """Test basic basis encoding (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Encode basis state |0⟩ (index 0 with 2 qubits)
    qtensor = engine.encode([0.0], 2, "basis")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.is_cuda
    assert torch_tensor.shape == (1, 4)  # 2^2 = 4 amplitudes

    # |0⟩ = [1, 0, 0, 0]
    expected = torch.tensor([[1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j]], device="cuda:0")
    assert torch.allclose(torch_tensor, expected.to(torch_tensor.dtype))


@pytest.mark.gpu
def test_basis_encode_nonzero_index():
    """Test basis encoding with non-zero index (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Encode basis state |3⟩ = |11⟩ (index 3 with 2 qubits)
    qtensor = engine.encode([3.0], 2, "basis")
    torch_tensor = torch.from_dlpack(qtensor)

    # |3⟩ = [0, 0, 0, 1]
    expected = torch.tensor([[0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j]], device="cuda:0")
    assert torch.allclose(torch_tensor, expected.to(torch_tensor.dtype))


@pytest.mark.gpu
def test_basis_encode_3_qubits():
    """Test basis encoding with 3 qubits (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Encode basis state |5⟩ = |101⟩ (index 5 with 3 qubits)
    qtensor = engine.encode([5.0], 3, "basis")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.shape == (1, 8)  # 2^3 = 8 amplitudes

    # |5⟩ should have amplitude 1 at index 5
    # Check that only index 5 is non-zero
    host_tensor = torch_tensor.cpu().squeeze()
    assert host_tensor[5].real == 1.0
    assert host_tensor[5].imag == 0.0
    for i in range(8):
        if i != 5:
            assert host_tensor[i].real == 0.0
            assert host_tensor[i].imag == 0.0


@pytest.mark.gpu
def test_basis_encode_errors():
    """Test error handling for basis encoding (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Test index out of bounds (2^2 = 4, so max index is 3)
    with pytest.raises(RuntimeError, match="exceeds state vector size"):
        engine.encode([4.0], 2, "basis")

    # Test negative index
    with pytest.raises(RuntimeError, match="non-negative"):
        engine.encode([-1.0], 2, "basis")

    # Test non-integer index
    with pytest.raises(RuntimeError, match="integer"):
        engine.encode([1.5], 2, "basis")

    # Test empty input
    with pytest.raises(RuntimeError, match="empty"):
        engine.encode([], 2, "basis")

    # Test multiple values (basis expects exactly 1)
    with pytest.raises(RuntimeError, match="expects exactly 1"):
        engine.encode([0.0, 1.0], 2, "basis")
