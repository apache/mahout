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
import torch

from .qdp_test_utils import requires_qdp


def _has_multi_gpu():
    """Check if multiple GPUs are available via PyTorch."""
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except ImportError:
        return False


@requires_qdp
def test_import():
    """Test that PyO3 bindings are properly imported."""
    import _qdp

    assert hasattr(_qdp, "QdpEngine")
    assert hasattr(_qdp, "QuantumTensor")

    # Test that QdpEngine has the new encode_from_tensorflow method
    from _qdp import QdpEngine

    assert hasattr(QdpEngine, "encode_from_tensorflow")
    assert callable(getattr(QdpEngine, "encode_from_tensorflow"))


@requires_qdp
@pytest.mark.gpu
def test_encode():
    """Test encoding returns QuantumTensor (requires GPU)."""
    from _qdp import QdpEngine, QuantumTensor

    engine = QdpEngine(0)
    data = [0.5, 0.5, 0.5, 0.5]
    qtensor = engine.encode(data, 2, "amplitude")
    assert isinstance(qtensor, QuantumTensor)


@requires_qdp
@pytest.mark.gpu
def test_dlpack_device():
    """Test __dlpack_device__ method (requires GPU)."""
    from _qdp import QdpEngine

    engine = QdpEngine(0)
    data = [1.0, 2.0, 3.0, 4.0]
    qtensor = engine.encode(data, 2, "amplitude")

    device_info = qtensor.__dlpack_device__()
    assert device_info == (2, 0), "Expected (2, 0) for CUDA device 0"


@requires_qdp
@pytest.mark.gpu
@pytest.mark.skipif(
    not _has_multi_gpu(), reason="Multi-GPU setup required for this test"
)
def test_dlpack_device_id_non_zero():
    """Test device_id propagation for non-zero devices (requires multi-GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_dlpack_single_use():
    """Test that __dlpack__ can only be called once (requires GPU)."""
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


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("stream", [1, 2], ids=["stream_legacy", "stream_per_thread"])
def test_dlpack_with_stream(stream):
    """Test __dlpack__(stream=...) syncs CUDA stream before returning capsule (DLPack 0.8+)."""
    from _qdp import QdpEngine

    engine = QdpEngine(0)
    data = [1.0, 2.0, 3.0, 4.0]
    qtensor = engine.encode(data, 2, "amplitude")

    # stream=1 (legacy default) or 2 (per-thread default) should sync and return capsule
    capsule = qtensor.__dlpack__(stream=stream)
    torch_tensor = torch.from_dlpack(capsule)
    assert torch_tensor.is_cuda
    assert torch_tensor.shape == (1, 4)


@requires_qdp
@pytest.mark.gpu
def test_pytorch_integration():
    """Test PyTorch integration via DLPack (requires GPU and PyTorch)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "precision,expected_dtype",
    [
        ("float32", "complex64"),
        ("float64", "complex128"),
    ],
)
def test_precision(precision, expected_dtype):
    """Test different precision settings produce correct output dtypes."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    engine = QdpEngine(0, precision=precision)
    data = [1.0, 2.0, 3.0, 4.0]
    qtensor = engine.encode(data, 2, "amplitude")

    torch_tensor = torch.from_dlpack(qtensor)
    expected = getattr(torch, expected_dtype)
    assert torch_tensor.dtype == expected, (
        f"Expected {expected_dtype}, got {torch_tensor.dtype}"
    )


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "data_shape,expected_shape",
    [
        ([1.0, 2.0, 3.0, 4.0], (1, 4)),  # 1D tensor -> single sample
        (
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            (3, 4),
        ),  # 2D tensor -> batch
    ],
)
def test_encode_tensor_cpu(data_shape, expected_shape):
    """Test encoding from CPU PyTorch tensor (1D or 2D, zero-copy)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)
    data = torch.tensor(data_shape, dtype=torch.float64)
    if len(data_shape) > 1:
        assert data.is_contiguous(), "Test tensor should be contiguous for zero-copy"

    qtensor = engine.encode(data, 2, "amplitude")

    # Verify result
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.is_cuda
    assert torch_tensor.shape == expected_shape


@requires_qdp
@pytest.mark.gpu
def test_encode_from_tensorflow_binding():
    """Test TensorFlow TensorProto binding path (requires GPU and TensorFlow)."""
    pytest.importorskip("torch")
    tf = pytest.importorskip("tensorflow")
    import numpy as np
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


@requires_qdp
@pytest.mark.gpu
def test_encode_errors():
    """Test error handling for unified encode method."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "data_shape,expected_shape,expected_batch_size",
    [
        ([1.0, 2.0, 3.0, 4.0], (1, 4), 1),  # 1D tensor -> single sample
        (
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            (3, 4),
            3,
        ),  # 2D tensor -> batch
    ],
)
def test_encode_cuda_tensor(data_shape, expected_shape, expected_batch_size):
    """Test encoding from CUDA tensor (1D or 2D, zero-copy)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create CUDA tensor
    data = torch.tensor(data_shape, dtype=torch.float64, device="cuda:0")
    qtensor = engine.encode(data, 2, "amplitude")

    # Verify result
    result = torch.from_dlpack(qtensor)
    assert result.is_cuda
    assert result.shape == expected_shape

    # Verify normalization (each sample should have unit norm)
    for i in range(expected_batch_size):
        norm = torch.sqrt(torch.sum(torch.abs(result[i]) ** 2))
        assert torch.isclose(norm, torch.tensor(1.0, device="cuda:0"), atol=1e-6)


@requires_qdp
@pytest.mark.gpu
def test_encode_cuda_tensor_wrong_dtype():
    """Test error when CUDA tensor has wrong dtype for amplitude (e.g. float16)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Amplitude encoding accepts float64 or float32 only; float16 is invalid
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16, device="cuda:0")
    with pytest.raises(RuntimeError, match="float64 or float32"):
        engine.encode(data, 2, "amplitude")


@requires_qdp
@pytest.mark.gpu
def test_encode_cuda_tensor_non_contiguous():
    """Test error when CUDA tensor is non-contiguous."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
@pytest.mark.skipif(
    not _has_multi_gpu(), reason="Multi-GPU setup required for this test"
)
def test_encode_cuda_tensor_device_mismatch():
    """Test error when CUDA tensor is on wrong device (multi-GPU only)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    # Engine on device 0
    engine = QdpEngine(0)

    # Tensor on device 1 (wrong device)
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device="cuda:1")
    with pytest.raises(RuntimeError, match="Device mismatch"):
        engine.encode(data, 2, "amplitude")


@requires_qdp
@pytest.mark.gpu
def test_encode_cuda_tensor_empty():
    """Test error when CUDA tensor is empty."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create empty CUDA tensor
    data = torch.tensor([], dtype=torch.float64, device="cuda:0")
    with pytest.raises(RuntimeError, match="CUDA tensor cannot be empty"):
        engine.encode(data, 2, "amplitude")


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "data_shape,is_batch",
    [
        ([1.0, 2.0, 3.0, 4.0], False),  # 1D tensor
        ([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], True),  # 2D tensor (batch)
    ],
)
def test_encode_cuda_tensor_preserves_input(data_shape, is_batch):
    """Test that input CUDA tensor (1D or 2D) is not modified after encoding."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create CUDA tensor and save a copy
    data = torch.tensor(data_shape, dtype=torch.float64, device="cuda:0")
    data_clone = data.clone()

    # Encode
    qtensor = engine.encode(data, 2, "amplitude")
    _ = torch.from_dlpack(qtensor)

    # Verify original tensor is unchanged
    assert torch.equal(data, data_clone)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("encoding_method", ["iqp"])
def test_encode_cuda_tensor_unsupported_encoding(encoding_method):
    """Test error when using CUDA tensor with an encoding not supported on GPU (only amplitude, angle, basis)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # CUDA path only supports amplitude, angle, basis; iqp/iqp-z should raise unsupported error
    data = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device="cuda:0")

    with pytest.raises(
        RuntimeError,
        match="only supports .*amplitude.*angle.*basis.*Use tensor.cpu",
    ):
        engine.encode(data, 2, encoding_method)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "input_type,error_match",
    [
        ("cuda_tensor", "Unsupported CUDA tensor shape: 3D"),
        ("cpu_tensor", "Unsupported tensor shape: 3D"),
        ("numpy_array", "Unsupported array shape: 3D"),
    ],
)
def test_encode_3d_rejected(input_type, error_match):
    """Test error when input has 3+ dimensions (CUDA tensor, CPU tensor, or NumPy array)."""
    pytest.importorskip("torch")
    import numpy as np
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Create 3D data based on input type
    if input_type == "cuda_tensor":
        data = torch.randn(2, 3, 4, dtype=torch.float64, device="cuda:0")
    elif input_type == "cpu_tensor":
        data = torch.randn(2, 3, 4, dtype=torch.float64)
    elif input_type == "numpy_array":
        data = np.random.randn(2, 3, 4).astype(np.float64)
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    with pytest.raises(RuntimeError, match=error_match):
        engine.encode(data, 2, "amplitude")


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "tensor_factory,description",
    [
        (lambda: torch.zeros(4, dtype=torch.float64, device="cuda:0"), "zeros"),
        (
            lambda: torch.tensor(
                [1.0, float("nan"), 3.0, 4.0], dtype=torch.float64, device="cuda:0"
            ),
            "NaN",
        ),
        (
            lambda: torch.tensor(
                [1.0, float("inf"), 3.0, 4.0], dtype=torch.float64, device="cuda:0"
            ),
            "Inf",
        ),
    ],
)
def test_encode_cuda_tensor_non_finite_values(tensor_factory, description):
    """Test error when CUDA tensor contains non-finite values (zeros, NaN, Inf)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)
    data = tensor_factory()

    with pytest.raises(RuntimeError, match="zero or non-finite norm"):
        engine.encode(data, 2, "amplitude")


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "precision,expected_dtype",
    [
        ("float32", torch.complex64),
        ("float64", torch.complex128),
    ],
)
def test_encode_cuda_tensor_output_dtype(precision, expected_dtype):
    """Test that CUDA tensor encoding produces correct output dtype."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0, precision=precision)
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device="cuda:0")
    result = torch.from_dlpack(engine.encode(data, 2, "amplitude"))
    assert result.dtype == expected_dtype, (
        f"Expected {expected_dtype}, got {result.dtype}"
    )


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "precision,expected_dtype",
    [
        ("float32", torch.complex64),
        ("float64", torch.complex128),
    ],
)
def test_encode_cuda_tensor_float32_input_output_dtype(precision, expected_dtype):
    """Test that 1D float32 CUDA amplitude encoding respects engine precision (f32 path)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0, precision=precision)
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda:0")
    result = torch.from_dlpack(engine.encode(data, 2, "amplitude"))
    assert result.dtype == expected_dtype, (
        f"Expected {expected_dtype}, got {result.dtype}"
    )


@requires_qdp
@pytest.mark.gpu
def test_basis_encode_basic():
    """Test basic basis encoding (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_basis_encode_nonzero_index():
    """Test basis encoding with non-zero index (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_basis_encode_3_qubits():
    """Test basis encoding with 3 qubits (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_basis_encode_errors():
    """Test error handling for basis encoding (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_angle_encode_basic():
    """Test basic angle encoding (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_angle_encode_nonzero_angles():
    """Test angle encoding with non-zero angles (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_angle_encode_batch():
    """Test batch angle encoding (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_angle_encode_errors():
    """Test error handling for angle encoding (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "data_shape,expected_shape",
    [
        ([1.0, 2.0, 3.0, 4.0], (1, 4)),  # 1D array -> single sample
        (
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            (2, 4),
        ),  # 2D array -> batch
    ],
)
def test_encode_numpy_array(data_shape, expected_shape):
    """Test encoding from NumPy array (1D or 2D)."""
    pytest.importorskip("torch")
    import numpy as np
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)
    data = np.array(data_shape, dtype=np.float64)
    qtensor = engine.encode(data, 2, "amplitude")

    # Verify result
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.is_cuda
    assert torch_tensor.shape == expected_shape


@requires_qdp
@pytest.mark.gpu
def test_encode_pathlib_path():
    """Test encoding from pathlib.Path object."""
    pytest.importorskip("torch")
    import numpy as np
    from pathlib import Path
    import tempfile
    import os
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)
    num_qubits = 2
    sample_size = 2**num_qubits

    # Create temporary .npy file
    data = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float64)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_path = Path(f.name)
        np.save(npy_path, data)

    try:
        # Test with pathlib.Path
        qtensor = engine.encode(npy_path, num_qubits, "amplitude")
        torch_tensor = torch.from_dlpack(qtensor)
        assert torch_tensor.is_cuda
        assert torch_tensor.shape == (2, sample_size)
    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)


@requires_qdp
@pytest.mark.gpu
def test_iqp_z_encode_basic():
    """Test basic IQP-Z encoding with zero angles (requires GPU).

    With zero parameters, IQP produces |00...0⟩ because:
    - H^n|0⟩^n gives uniform superposition
    - Zero phases leave state unchanged
    - H^n transforms back to |0⟩^n
    """
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_iqp_z_encode_nonzero():
    """Test IQP-Z encoding with non-zero angles (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_iqp_encode_basic():
    """Test basic IQP encoding with ZZ interactions (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_iqp_encode_zz_effect():
    """Test that ZZ interaction produces different result than Z-only (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_iqp_encode_3_qubits():
    """Test IQP encoding with 3 qubits (requires GPU)."""
    pytest.importorskip("torch")
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
    row = [1.0 + 0j] + [0.0 + 0j] * 7
    expected = torch.tensor([row], dtype=torch.complex128, device="cuda:0")
    assert torch.allclose(torch_tensor, expected, atol=1e-6)


@requires_qdp
@pytest.mark.gpu
def test_iqp_z_encode_batch():
    """Test batch IQP-Z encoding (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_iqp_encode_batch():
    """Test batch IQP encoding with ZZ interactions (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_iqp_encode_single_qubit():
    """Test IQP encoding with single qubit edge case (requires GPU)."""
    pytest.importorskip("torch")
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


@requires_qdp
@pytest.mark.gpu
def test_iqp_encode_errors():
    """Test error handling for IQP encoding (requires GPU)."""
    pytest.importorskip("torch")
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


# ==================== IQP FWT Optimization Tests ====================


@pytest.mark.gpu
def test_iqp_fwt_normalization():
    """Test that FWT-optimized IQP produces normalized states (requires GPU)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Test across FWT threshold (FWT_MIN_QUBITS = 4)
    for num_qubits in [3, 4, 5, 6, 7, 8]:
        # Full IQP (n + n*(n-1)/2 parameters)
        data_len = num_qubits + num_qubits * (num_qubits - 1) // 2
        data = [0.1 * i for i in range(data_len)]

        qtensor = engine.encode(data, num_qubits, "iqp")
        torch_tensor = torch.from_dlpack(qtensor)

        # Verify normalization (sum of |amplitude|^2 = 1)
        norm = torch.sum(torch.abs(torch_tensor) ** 2)
        assert torch.isclose(norm, torch.tensor(1.0, device="cuda:0"), atol=1e-6), (
            f"IQP {num_qubits} qubits not normalized: got {norm.item()}"
        )


@pytest.mark.gpu
def test_iqp_z_fwt_normalization():
    """Test that FWT-optimized IQP-Z produces normalized states (requires GPU)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Test across FWT threshold
    for num_qubits in [3, 4, 5, 6, 7, 8]:
        data = [0.2 * i for i in range(num_qubits)]

        qtensor = engine.encode(data, num_qubits, "iqp-z")
        torch_tensor = torch.from_dlpack(qtensor)

        norm = torch.sum(torch.abs(torch_tensor) ** 2)
        assert torch.isclose(norm, torch.tensor(1.0, device="cuda:0"), atol=1e-6), (
            f"IQP-Z {num_qubits} qubits not normalized: got {norm.item()}"
        )


@pytest.mark.gpu
def test_iqp_fwt_zero_params_gives_zero_state():
    """Test that zero parameters produce |0...0⟩ state (requires GPU).

    With zero parameters, the IQP circuit is H^n * I * H^n = I,
    so |0⟩^n maps to |0⟩^n with amplitude 1 at index 0.
    """
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Test FWT-optimized path (n >= 4)
    for num_qubits in [4, 5, 6]:
        data_len = num_qubits + num_qubits * (num_qubits - 1) // 2
        data = [0.0] * data_len

        qtensor = engine.encode(data, num_qubits, "iqp")
        torch_tensor = torch.from_dlpack(qtensor)

        # Should get |0...0⟩: amplitude 1 at index 0, 0 elsewhere
        state_len = 1 << num_qubits
        row = [1.0 + 0j] + [0.0 + 0j] * (state_len - 1)
        expected = torch.tensor([row], dtype=torch_tensor.dtype, device="cuda:0")

        assert torch.allclose(torch_tensor, expected, atol=1e-6), (
            f"IQP {num_qubits} qubits with zero params should give |0⟩ state"
        )


@pytest.mark.gpu
def test_iqp_fwt_batch_normalization():
    """Test that FWT-optimized batch IQP produces normalized states (requires GPU)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Test batch encoding across FWT threshold
    for num_qubits in [4, 5, 6]:
        data_len = num_qubits + num_qubits * (num_qubits - 1) // 2
        batch_size = 8

        data = torch.tensor(
            [
                [0.1 * (i + j * data_len) for i in range(data_len)]
                for j in range(batch_size)
            ],
            dtype=torch.float64,
        )

        qtensor = engine.encode(data, num_qubits, "iqp")
        torch_tensor = torch.from_dlpack(qtensor)

        assert torch_tensor.shape == (batch_size, 1 << num_qubits)

        # Check each sample is normalized
        for i in range(batch_size):
            norm = torch.sum(torch.abs(torch_tensor[i]) ** 2)
            assert torch.isclose(norm, torch.tensor(1.0, device="cuda:0"), atol=1e-6), (
                f"IQP batch sample {i} not normalized: got {norm.item()}"
            )


@pytest.mark.gpu
def test_iqp_fwt_deterministic():
    """Test that FWT-optimized IQP is deterministic (requires GPU)."""
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    num_qubits = 6  # Uses FWT path
    data_len = num_qubits + num_qubits * (num_qubits - 1) // 2
    data = [0.3 * i for i in range(data_len)]

    # Run encoding twice
    qtensor1 = engine.encode(data, num_qubits, "iqp")
    tensor1 = torch.from_dlpack(qtensor1).clone()

    qtensor2 = engine.encode(data, num_qubits, "iqp")
    tensor2 = torch.from_dlpack(qtensor2)

    # Results should be identical
    assert torch.allclose(tensor1, tensor2, atol=1e-10), (
        "IQP FWT encoding should be deterministic"
    )


@pytest.mark.gpu
def test_iqp_fwt_shared_vs_global_memory_threshold():
    """Test IQP encoding at shared memory threshold boundary (requires GPU).

    FWT_SHARED_MEM_THRESHOLD = 10, so:
    - n <= 10: uses shared memory FWT
    - n > 10: uses global memory FWT
    """
    pytest.importorskip("torch")
    from _qdp import QdpEngine

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    engine = QdpEngine(0)

    # Test at and around the shared memory threshold
    # n <= 10: shared memory FWT, n > 10: global memory FWT (multi-launch)
    for num_qubits in [9, 10, 11]:
        data_len = num_qubits + num_qubits * (num_qubits - 1) // 2
        data = [0.05 * i for i in range(data_len)]

        qtensor = engine.encode(data, num_qubits, "iqp")
        torch_tensor = torch.from_dlpack(qtensor)

        assert torch_tensor.shape == (1, 1 << num_qubits)

        norm = torch.sum(torch.abs(torch_tensor) ** 2)
        assert torch.isclose(norm, torch.tensor(1.0, device="cuda:0"), atol=1e-6), (
            f"IQP {num_qubits} qubits not normalized at threshold: got {norm.item()}"
        )
