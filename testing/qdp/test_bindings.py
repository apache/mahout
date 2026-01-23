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


@pytest.mark.gpu
def test_encode_cuda_tensor_1d():
    """Test encoding from 1D CUDA tensor (single sample, zero-copy)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create 1D CUDA tensor
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device="cuda:0")
    qtensor = engine.encode(data, 2, "amplitude")

    # Verify result
    result = torch.from_dlpack(qtensor)
    assert result.is_cuda
    assert result.shape == (1, 4)  # 2^2 = 4 amplitudes

    # Verify normalization (amplitudes should have unit norm)
    norm = torch.sqrt(torch.sum(torch.abs(result) ** 2))
    assert torch.isclose(norm, torch.tensor(1.0, device="cuda:0"), atol=1e-6)


@pytest.mark.gpu
def test_encode_cuda_tensor_2d_batch():
    """Test encoding from 2D CUDA tensor (batch, zero-copy)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create 2D CUDA tensor (batch_size=3, features=4)
    data = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=torch.float64,
        device="cuda:0",
    )
    qtensor = engine.encode(data, 2, "amplitude")

    # Verify result
    result = torch.from_dlpack(qtensor)
    assert result.is_cuda
    assert result.shape == (3, 4)  # batch_size=3, 2^2=4

    # Verify each sample is normalized
    for i in range(3):
        norm = torch.sqrt(torch.sum(torch.abs(result[i]) ** 2))
        assert torch.isclose(norm, torch.tensor(1.0, device="cuda:0"), atol=1e-6)


@pytest.mark.gpu
def test_encode_cuda_tensor_wrong_dtype():
    """Test error when CUDA tensor has wrong dtype (non-float64)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create CUDA tensor with float32 dtype (wrong)
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda:0")
    with pytest.raises(RuntimeError, match="CUDA tensor must have dtype float64"):
        engine.encode(data, 2, "amplitude")


@pytest.mark.gpu
def test_encode_cuda_tensor_non_contiguous():
    """Test error when CUDA tensor is non-contiguous."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create non-contiguous CUDA tensor (via transpose)
    data = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, device="cuda:0"
    ).t()
    assert not data.is_contiguous()

    with pytest.raises(RuntimeError, match="CUDA tensor must be contiguous"):
        engine.encode(data, 2, "amplitude")


@pytest.mark.gpu
@pytest.mark.skipif(
    not _has_multi_gpu(), reason="Multi-GPU setup required for this test"
)
def test_encode_cuda_tensor_device_mismatch():
    """Test error when CUDA tensor is on wrong device (multi-GPU only)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    # Engine on device 0
    engine = QdpEngine(0)

    # Tensor on device 1 (wrong device)
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device="cuda:1")
    with pytest.raises(RuntimeError, match="Device mismatch"):
        engine.encode(data, 2, "amplitude")


@pytest.mark.gpu
def test_encode_cuda_tensor_empty():
    """Test error when CUDA tensor is empty."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create empty CUDA tensor
    data = torch.tensor([], dtype=torch.float64, device="cuda:0")
    with pytest.raises(RuntimeError, match="CUDA tensor cannot be empty"):
        engine.encode(data, 2, "amplitude")


@pytest.mark.gpu
def test_encode_cuda_tensor_preserves_input():
    """Test that input CUDA tensor is not modified after encoding."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create CUDA tensor and save a copy
    original_data = [1.0, 2.0, 3.0, 4.0]
    data = torch.tensor(original_data, dtype=torch.float64, device="cuda:0")
    data_clone = data.clone()

    # Encode
    qtensor = engine.encode(data, 2, "amplitude")
    _ = torch.from_dlpack(qtensor)

    # Verify original tensor is unchanged
    assert torch.equal(data, data_clone)


@pytest.mark.gpu
def test_encode_cuda_tensor_unsupported_encoding():
    """Test error when using CUDA tensor with unsupported encoding method."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # CUDA tensors currently only support amplitude encoding
    # Use non-zero data to avoid normalization issues
    data = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device="cuda:0")

    with pytest.raises(RuntimeError, match="only supports 'amplitude' method"):
        engine.encode(data, 2, "basis")

    with pytest.raises(RuntimeError, match="only supports 'amplitude' method"):
        engine.encode(data, 2, "angle")


@pytest.mark.gpu
def test_encode_cuda_tensor_3d_rejected():
    """Test error when CUDA tensor has 3+ dimensions."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create 3D CUDA tensor (should be rejected)
    data = torch.randn(2, 3, 4, dtype=torch.float64, device="cuda:0")
    with pytest.raises(RuntimeError, match="Unsupported CUDA tensor shape: 3D"):
        engine.encode(data, 2, "amplitude")


@pytest.mark.gpu
def test_encode_cuda_tensor_zero_values():
    """Test error when CUDA tensor contains all zeros (zero norm)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create CUDA tensor with all zeros (cannot be normalized)
    data = torch.zeros(4, dtype=torch.float64, device="cuda:0")
    with pytest.raises(RuntimeError, match="zero or non-finite norm"):
        engine.encode(data, 2, "amplitude")


@pytest.mark.gpu
def test_encode_cuda_tensor_nan_values():
    """Test error when CUDA tensor contains NaN values."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create CUDA tensor with NaN
    data = torch.tensor(
        [1.0, float("nan"), 3.0, 4.0], dtype=torch.float64, device="cuda:0"
    )
    with pytest.raises(RuntimeError, match="zero or non-finite norm"):
        engine.encode(data, 2, "amplitude")


@pytest.mark.gpu
def test_encode_cuda_tensor_inf_values():
    """Test error when CUDA tensor contains Inf values."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create CUDA tensor with Inf
    data = torch.tensor(
        [1.0, float("inf"), 3.0, 4.0], dtype=torch.float64, device="cuda:0"
    )
    with pytest.raises(RuntimeError, match="zero or non-finite norm"):
        engine.encode(data, 2, "amplitude")


@pytest.mark.gpu
def test_encode_cuda_tensor_output_dtype():
    """Test that CUDA tensor encoding produces correct output dtype."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    # Test default precision (float32 -> complex64)
    engine_f32 = QdpEngine(0, precision="float32")
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device="cuda:0")
    result = torch.from_dlpack(engine_f32.encode(data, 2, "amplitude"))
    assert result.dtype == torch.complex64, f"Expected complex64, got {result.dtype}"

    # Test float64 precision (float64 -> complex128)
    engine_f64 = QdpEngine(0, precision="float64")
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device="cuda:0")
    result = torch.from_dlpack(engine_f64.encode(data, 2, "amplitude"))
    assert result.dtype == torch.complex128, f"Expected complex128, got {result.dtype}"


@pytest.mark.gpu
def test_encode_cuda_tensor_preserves_input_batch():
    """Test that input 2D CUDA tensor (batch) is not modified after encoding."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create 2D CUDA tensor and save a copy
    data = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=torch.float64,
        device="cuda:0",
    )
    data_clone = data.clone()

    # Encode
    qtensor = engine.encode(data, 2, "amplitude")
    _ = torch.from_dlpack(qtensor)

    # Verify original tensor is unchanged
    assert torch.equal(data, data_clone)


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


@pytest.mark.gpu
def test_angle_encode_basic():
    """Test basic angle encoding (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Angles [0, 0] should map to |00> with amplitude 1 at index 0.
    qtensor = engine.encode([0.0, 0.0], 2, "angle")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.is_cuda
    assert torch_tensor.shape == (1, 4)

    expected = torch.tensor([[1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j]], device="cuda:0")
    assert torch.allclose(torch_tensor, expected.to(torch_tensor.dtype))


@pytest.mark.gpu
def test_angle_encode_nonzero_angles():
    """Test angle encoding with non-zero angles (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    angles = [torch.pi / 2, 0.0]
    qtensor = engine.encode(angles, 2, "angle")
    torch_tensor = torch.from_dlpack(qtensor)

    expected = torch.tensor([[0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j]], device="cuda:0")
    assert torch.allclose(
        torch_tensor, expected.to(torch_tensor.dtype), atol=1e-6, rtol=1e-6
    )


@pytest.mark.gpu
def test_angle_encode_batch():
    """Test batch angle encoding (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    data = torch.tensor([[0.0, 0.0], [torch.pi / 2, 0.0]], dtype=torch.float64)
    qtensor = engine.encode(data, 2, "angle")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.shape == (2, 4)

    expected = torch.tensor(
        [
            [1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j],
        ],
        device="cuda:0",
    )
    assert torch.allclose(
        torch_tensor, expected.to(torch_tensor.dtype), atol=1e-6, rtol=1e-6
    )


@pytest.mark.gpu
def test_angle_encode_errors():
    """Test error handling for angle encoding (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Wrong length (expects one angle per qubit)
    with pytest.raises(RuntimeError, match="expects 2 values"):
        engine.encode([0.0], 2, "angle")

    # Non-finite angle
    with pytest.raises(RuntimeError, match="must be finite"):
        engine.encode([float("nan"), 0.0], 2, "angle")


# ==================== IQP Encoding Tests ====================


@pytest.mark.gpu
def test_iqp_z_encode_basic():
    """Test basic IQP-Z encoding with zero angles (requires GPU).

    With zero parameters, IQP produces |00...0⟩ because:
    - H^n|0⟩^n gives uniform superposition
    - Zero phases leave state unchanged
    - H^n transforms back to |0⟩^n
    """
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # With zero angles, H^n * I * H^n |0⟩ = |0⟩, so amplitude 1 at index 0
    qtensor = engine.encode([0.0, 0.0], 2, "iqp-z")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.is_cuda
    assert torch_tensor.shape == (1, 4)

    # Should get |00⟩ state: amplitude 1 at index 0, 0 elsewhere
    expected = torch.tensor([[1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j]], device="cuda:0")
    assert torch.allclose(torch_tensor, expected, atol=1e-6)


@pytest.mark.gpu
def test_iqp_z_encode_nonzero():
    """Test IQP-Z encoding with non-zero angles (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # With non-zero angles, we get interference patterns
    # Using pi on qubit 0: phase flip when qubit 0 is |1⟩
    qtensor = engine.encode([torch.pi, 0.0], 2, "iqp-z")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.shape == (1, 4)

    # The state should be different from |00⟩
    # Verify normalization (sum of |amplitude|^2 = 1)
    norm = torch.sum(torch.abs(torch_tensor) ** 2)
    assert torch.allclose(norm, torch.tensor(1.0, device="cuda:0"), atol=1e-6)


@pytest.mark.gpu
def test_iqp_encode_basic():
    """Test basic IQP encoding with ZZ interactions (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # 2 qubits needs 3 parameters: [theta_0, theta_1, J_01]
    # With all zeros, should get |00⟩ state
    qtensor = engine.encode([0.0, 0.0, 0.0], 2, "iqp")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.is_cuda
    assert torch_tensor.shape == (1, 4)

    # Should get |00⟩ state: amplitude 1 at index 0, 0 elsewhere
    expected = torch.tensor([[1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j]], device="cuda:0")
    assert torch.allclose(torch_tensor, expected, atol=1e-6)


@pytest.mark.gpu
def test_iqp_encode_zz_effect():
    """Test that ZZ interaction produces different result than Z-only (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Same single-qubit angles, but with ZZ interaction
    angles_z_only = [torch.pi / 4, torch.pi / 4]
    angles_with_zz = [torch.pi / 4, torch.pi / 4, torch.pi / 2]  # Add J_01

    qtensor_z = engine.encode(angles_z_only, 2, "iqp-z")
    qtensor_zz = engine.encode(angles_with_zz, 2, "iqp")

    tensor_z = torch.from_dlpack(qtensor_z)
    tensor_zz = torch.from_dlpack(qtensor_zz)

    # The two should be different due to ZZ interaction
    assert not torch.allclose(tensor_z, tensor_zz, atol=1e-6)

    # Both should be normalized
    norm_z = torch.sum(torch.abs(tensor_z) ** 2)
    norm_zz = torch.sum(torch.abs(tensor_zz) ** 2)
    assert torch.allclose(norm_z, torch.tensor(1.0, device="cuda:0"), atol=1e-6)
    assert torch.allclose(norm_zz, torch.tensor(1.0, device="cuda:0"), atol=1e-6)


@pytest.mark.gpu
def test_iqp_encode_3_qubits():
    """Test IQP encoding with 3 qubits (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0, precision="float64")

    # 3 qubits needs 6 parameters: [theta_0, theta_1, theta_2, J_01, J_02, J_12]
    # With all zeros, should get |000⟩ state
    qtensor = engine.encode([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, "iqp")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.shape == (1, 8)

    # Should get |000⟩ state: amplitude 1 at index 0, 0 elsewhere
    expected = torch.zeros((1, 8), dtype=torch.complex128, device="cuda:0")
    expected[0, 0] = 1.0 + 0j
    assert torch.allclose(torch_tensor, expected, atol=1e-6)


@pytest.mark.gpu
def test_iqp_z_encode_batch():
    """Test batch IQP-Z encoding (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Batch of 2 samples with different angles
    data = torch.tensor([[0.0, 0.0], [torch.pi, 0.0]], dtype=torch.float64)
    qtensor = engine.encode(data, 2, "iqp-z")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.shape == (2, 4)

    # First sample (zero angles) should give |00⟩
    expected_0 = torch.tensor([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], device="cuda:0")
    assert torch.allclose(torch_tensor[0], expected_0, atol=1e-6)

    # Second sample should be different and normalized
    norm_1 = torch.sum(torch.abs(torch_tensor[1]) ** 2)
    assert torch.allclose(norm_1, torch.tensor(1.0, device="cuda:0"), atol=1e-6)


@pytest.mark.gpu
def test_iqp_encode_batch():
    """Test batch IQP encoding with ZZ interactions (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Batch of 2 samples, each with 3 parameters (2 qubits)
    data = torch.tensor(
        [[0.0, 0.0, 0.0], [torch.pi / 4, torch.pi / 4, torch.pi / 2]],
        dtype=torch.float64,
    )
    qtensor = engine.encode(data, 2, "iqp")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.shape == (2, 4)

    # First sample (zero params) should give |00⟩
    expected_0 = torch.tensor([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], device="cuda:0")
    assert torch.allclose(torch_tensor[0], expected_0, atol=1e-6)

    # Second sample should be different and normalized
    norm_1 = torch.sum(torch.abs(torch_tensor[1]) ** 2)
    assert torch.allclose(norm_1, torch.tensor(1.0, device="cuda:0"), atol=1e-6)


@pytest.mark.gpu
def test_iqp_encode_single_qubit():
    """Test IQP encoding with single qubit edge case (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # 1 qubit, iqp-z needs 1 parameter
    qtensor = engine.encode([0.0], 1, "iqp-z")
    torch_tensor = torch.from_dlpack(qtensor)

    assert torch_tensor.shape == (1, 2)

    # Zero angle gives |0⟩
    expected = torch.tensor([[1.0 + 0j, 0.0 + 0j]], device="cuda:0")
    assert torch.allclose(torch_tensor, expected, atol=1e-6)

    # 1 qubit, iqp needs 1 parameter (no pairs)
    qtensor2 = engine.encode([0.0], 1, "iqp")
    torch_tensor2 = torch.from_dlpack(qtensor2)
    assert torch.allclose(torch_tensor2, expected, atol=1e-6)


@pytest.mark.gpu
def test_iqp_encode_errors():
    """Test error handling for IQP encoding (requires GPU)."""
    pytest.importorskip("torch")
    import torch
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Wrong length for iqp-z (expects 2 for 2 qubits, got 3)
    with pytest.raises(RuntimeError, match="expects 2 values"):
        engine.encode([0.0, 0.0, 0.0], 2, "iqp-z")

    # Wrong length for iqp (expects 3 for 2 qubits, got 2)
    with pytest.raises(RuntimeError, match="expects 3 values"):
        engine.encode([0.0, 0.0], 2, "iqp")

    # Non-finite parameter (NaN)
    with pytest.raises(RuntimeError, match="must be finite"):
        engine.encode([float("nan"), 0.0], 2, "iqp-z")

    # Non-finite parameter (positive infinity)
    with pytest.raises(RuntimeError, match="must be finite"):
        engine.encode([0.0, float("inf"), 0.0], 2, "iqp")

    # Non-finite parameter (negative infinity)
    with pytest.raises(RuntimeError, match="must be finite"):
        engine.encode([float("-inf"), 0.0], 2, "iqp-z")
