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

"""
Tests for IQP (Instantaneous Quantum Polynomial) encoding via Python bindings.

IQP encoding creates quantum states:
|psi(x)> = 1/sqrt(2^n) sum_z exp(i*phi(z,x)) |z>

where phi(z,x) depends on the entanglement pattern:
- iqp (no entanglement): phi = sum_i x_i*z_i
- iqp_linear: phi = sum_i x_i*z_i + sum_i x_i*x_{i+1}*z_i*z_{i+1}
- iqp_full: phi = sum_i x_i*z_i + sum_{i<j} x_i*x_j*z_i*z_j
"""

import math

import numpy as np
import pytest
import torch

from _qdp import QdpEngine


# Tolerances for floating point comparisons
FLOAT64_TOLERANCE = 1e-7
FLOAT32_TOLERANCE = 1e-5


def iqp_reference_none(data: np.ndarray, num_qubits: int) -> np.ndarray:
    """CPU reference implementation for IQP encoding (no entanglement)."""
    state_len = 1 << num_qubits
    norm_factor = 1.0 / math.sqrt(state_len)
    num_features = min(len(data), num_qubits)

    result = np.zeros(state_len, dtype=np.complex128)
    for z in range(state_len):
        phase = 0.0
        for i in range(num_features):
            if (z >> i) & 1:
                phase += data[i]
        result[z] = norm_factor * (math.cos(phase) + 1j * math.sin(phase))

    return result


def iqp_reference_linear(data: np.ndarray, num_qubits: int) -> np.ndarray:
    """CPU reference implementation for IQP encoding (linear entanglement)."""
    state_len = 1 << num_qubits
    norm_factor = 1.0 / math.sqrt(state_len)
    num_features = min(len(data), num_qubits)

    result = np.zeros(state_len, dtype=np.complex128)
    for z in range(state_len):
        phase = 0.0
        # Single-qubit terms
        for i in range(num_features):
            if (z >> i) & 1:
                phase += data[i]
        # Linear entanglement terms
        for i in range(num_features - 1):
            bit_i = (z >> i) & 1
            bit_i1 = (z >> (i + 1)) & 1
            if bit_i and bit_i1:
                phase += data[i] * data[i + 1]
        result[z] = norm_factor * (math.cos(phase) + 1j * math.sin(phase))

    return result


def iqp_reference_full(data: np.ndarray, num_qubits: int) -> np.ndarray:
    """CPU reference implementation for IQP encoding (full entanglement)."""
    state_len = 1 << num_qubits
    norm_factor = 1.0 / math.sqrt(state_len)
    num_features = min(len(data), num_qubits)

    result = np.zeros(state_len, dtype=np.complex128)
    for z in range(state_len):
        phase = 0.0
        # Single-qubit terms
        for i in range(num_features):
            if (z >> i) & 1:
                phase += data[i]
        # Full entanglement terms
        for i in range(num_features):
            bit_i = (z >> i) & 1
            if not bit_i:
                continue
            for j in range(i + 1, num_features):
                bit_j = (z >> j) & 1
                if bit_j:
                    phase += data[i] * data[j]
        result[z] = norm_factor * (math.cos(phase) + 1j * math.sin(phase))

    return result


@pytest.fixture(scope="module")
def engine_float64():
    """Float64 precision engine for IQP tests."""
    return QdpEngine(device_id=0, precision="float64")


@pytest.fixture(scope="module")
def engine_float32():
    """Float32 precision engine for IQP tests."""
    return QdpEngine(device_id=0, precision="float32")


@pytest.mark.gpu
class TestIqpEncodingBasic:
    """Basic IQP encoding tests for all entanglement patterns."""

    def test_iqp_no_entanglement(self, engine_float64):
        """Test basic IQP encoding with no entanglement."""
        data = [0.5, 1.0]
        num_qubits = 2

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        reference = iqp_reference_none(np.array(data), num_qubits)

        np.testing.assert_allclose(
            gpu_np, reference, rtol=0, atol=FLOAT64_TOLERANCE,
            err_msg="IQP encoding (no entanglement) mismatch"
        )

        # Check normalization
        total_prob = np.sum(np.abs(gpu_np) ** 2)
        assert abs(total_prob - 1.0) < FLOAT64_TOLERANCE, (
            f"State not normalized: prob = {total_prob}"
        )

    def test_iqp_linear_entanglement(self, engine_float64):
        """Test IQP encoding with linear entanglement."""
        data = [0.3, 0.5, 0.7]
        num_qubits = 3

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp_linear")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        reference = iqp_reference_linear(np.array(data), num_qubits)

        np.testing.assert_allclose(
            gpu_np, reference, rtol=0, atol=FLOAT64_TOLERANCE,
            err_msg="IQP encoding (linear entanglement) mismatch"
        )

    def test_iqp_full_entanglement(self, engine_float64):
        """Test IQP encoding with full entanglement."""
        data = [0.2, 0.4, 0.6]
        num_qubits = 3

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp_full")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        reference = iqp_reference_full(np.array(data), num_qubits)

        np.testing.assert_allclose(
            gpu_np, reference, rtol=0, atol=FLOAT64_TOLERANCE,
            err_msg="IQP encoding (full entanglement) mismatch"
        )


@pytest.mark.gpu
class TestIqpEncodingInputTypes:
    """Test IQP encoding with different input types."""

    def test_numpy_array_input(self, engine_float64):
        """Test IQP encoding with NumPy array input."""
        data = np.array([0.1, 0.2, 0.3, 0.4])
        num_qubits = 4

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        reference = iqp_reference_none(data, num_qubits)

        np.testing.assert_allclose(
            gpu_np, reference, rtol=0, atol=FLOAT64_TOLERANCE,
            err_msg="IQP encoding with NumPy input mismatch"
        )

    def test_pytorch_tensor_input(self, engine_float64):
        """Test IQP encoding with PyTorch tensor input."""
        data = torch.tensor([0.5, 1.0, 0.25], dtype=torch.float64)
        num_qubits = 3

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        reference = iqp_reference_none(data.numpy(), num_qubits)

        np.testing.assert_allclose(
            gpu_np, reference, rtol=0, atol=FLOAT64_TOLERANCE,
            err_msg="IQP encoding with PyTorch input mismatch"
        )


@pytest.mark.gpu
class TestIqpBatchEncoding:
    """Test IQP batch encoding functionality."""

    def test_batch_encoding_no_entanglement(self, engine_float64):
        """Test IQP batch encoding with 2D NumPy array."""
        batch_data = np.array([
            [0.1, 0.2],
            [0.5, 0.6],
            [1.0, 1.5],
        ])
        num_qubits = 2
        state_len = 1 << num_qubits

        qtensor = engine_float64.encode(batch_data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)

        assert gpu_result.shape == (3, state_len), (
            f"Expected shape (3, {state_len}), got {gpu_result.shape}"
        )

        gpu_np = gpu_result.cpu().numpy()

        for i in range(3):
            reference = iqp_reference_none(batch_data[i], num_qubits)
            np.testing.assert_allclose(
                gpu_np[i], reference, rtol=0, atol=FLOAT64_TOLERANCE,
                err_msg=f"Batch sample {i} mismatch"
            )

    def test_batch_encoding_full_entanglement(self, engine_float64):
        """Test IQP batch encoding with full entanglement."""
        batch_data = np.array([
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7],
        ])
        num_qubits = 3
        state_len = 1 << num_qubits

        qtensor = engine_float64.encode(
            batch_data, num_qubits, encoding_method="iqp_full"
        )
        gpu_result = torch.from_dlpack(qtensor)

        assert gpu_result.shape == (2, state_len), (
            f"Expected shape (2, {state_len}), got {gpu_result.shape}"
        )

        gpu_np = gpu_result.cpu().numpy()

        for i in range(2):
            reference = iqp_reference_full(batch_data[i], num_qubits)
            np.testing.assert_allclose(
                gpu_np[i], reference, rtol=0, atol=FLOAT64_TOLERANCE,
                err_msg=f"Batch sample {i} mismatch"
            )


@pytest.mark.gpu
class TestIqpEncodingEdgeCases:
    """Test IQP encoding edge cases."""

    def test_zero_input_uniform_superposition(self, engine_float64):
        """Test IQP encoding with zero input (should produce uniform superposition)."""
        data = [0.0, 0.0, 0.0]
        num_qubits = 3
        state_len = 1 << num_qubits

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        # With zero input, all phases are 0, so all amplitudes = 1/sqrt(2^n)
        expected_amplitude = 1.0 / math.sqrt(state_len)

        for i in range(state_len):
            assert abs(gpu_np[i].real - expected_amplitude) < FLOAT64_TOLERANCE, (
                f"Element {i} real part should be {expected_amplitude}"
            )
            assert abs(gpu_np[i].imag) < FLOAT64_TOLERANCE, (
                f"Element {i} imag part should be 0"
            )

    def test_single_qubit_analytical(self, engine_float64):
        """Test IQP encoding with single qubit against analytical result."""
        data = [math.pi / 2]
        num_qubits = 1

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        norm = 1.0 / math.sqrt(2)

        # |0>: phase = 0, amplitude = 1/sqrt(2)
        assert abs(gpu_np[0].real - norm) < FLOAT64_TOLERANCE
        assert abs(gpu_np[0].imag) < FLOAT64_TOLERANCE

        # |1>: phase = pi/2, amplitude = 1/sqrt(2) * exp(i*pi/2) = i/sqrt(2)
        assert abs(gpu_np[1].real) < FLOAT64_TOLERANCE
        assert abs(gpu_np[1].imag - norm) < FLOAT64_TOLERANCE

    def test_large_state_10_qubits(self, engine_float64):
        """Test IQP encoding with larger state vector (10 qubits)."""
        num_qubits = 10
        num_features = 10
        data = np.array([i * 0.1 for i in range(num_features)])

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp_linear")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        state_len = 1 << num_qubits
        assert len(gpu_np) == state_len

        # Check normalization
        total_prob = np.sum(np.abs(gpu_np) ** 2)
        assert abs(total_prob - 1.0) < FLOAT64_TOLERANCE, (
            f"State not normalized: prob = {total_prob}"
        )

        # Full comparison against reference
        reference = iqp_reference_linear(data, num_qubits)
        np.testing.assert_allclose(
            gpu_np, reference, rtol=0, atol=FLOAT64_TOLERANCE,
            err_msg="Large state IQP encoding mismatch"
        )

    def test_negative_input_values(self, engine_float64):
        """Test IQP encoding with negative input values."""
        data = np.array([-0.5, 1.0, -0.25])
        num_qubits = 3

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        reference = iqp_reference_none(data, num_qubits)

        np.testing.assert_allclose(
            gpu_np, reference, rtol=0, atol=FLOAT64_TOLERANCE,
            err_msg="IQP encoding with negative values mismatch"
        )

    def test_features_more_than_qubits(self, engine_float64):
        """Test IQP encoding when num_features > num_qubits (truncation)."""
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 5 features
        num_qubits = 3  # Only 3 qubits

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        # Reference should also truncate to num_qubits features
        reference = iqp_reference_none(data, num_qubits)

        np.testing.assert_allclose(
            gpu_np, reference, rtol=0, atol=FLOAT64_TOLERANCE,
            err_msg="IQP encoding with features > qubits mismatch"
        )

    def test_features_fewer_than_qubits(self, engine_float64):
        """Test IQP encoding when num_features < num_qubits."""
        data = np.array([0.5, 0.7])  # 2 features
        num_qubits = 4  # 4 qubits

        qtensor = engine_float64.encode(data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)
        gpu_np = gpu_result.cpu().numpy().flatten()

        reference = iqp_reference_none(data, num_qubits)

        np.testing.assert_allclose(
            gpu_np, reference, rtol=0, atol=FLOAT64_TOLERANCE,
            err_msg="IQP encoding with features < qubits mismatch"
        )


@pytest.mark.gpu
class TestIqpEncodingPrecision:
    """Test IQP encoding with different output precisions."""

    def test_float32_output(self, engine_float32):
        """Test IQP encoding with float32 output precision."""
        data = [0.3, 0.6, 0.9]
        num_qubits = 3

        qtensor = engine_float32.encode(data, num_qubits, encoding_method="iqp")
        gpu_result = torch.from_dlpack(qtensor)

        assert gpu_result.dtype == torch.complex64, (
            f"Expected complex64, got {gpu_result.dtype}"
        )

        # Check normalization with lower precision tolerance
        total_prob = torch.sum(torch.abs(gpu_result) ** 2).item()
        assert abs(total_prob - 1.0) < FLOAT32_TOLERANCE, (
            f"State not normalized: prob = {total_prob}"
        )


@pytest.mark.gpu
class TestIqpEncodingErrors:
    """Test IQP encoding error handling."""

    def test_invalid_encoding_method(self, engine_float64):
        """Test error handling for invalid encoding method."""
        data = [0.5, 0.5]
        num_qubits = 2

        with pytest.raises(RuntimeError, match="[Uu]nknown|[Ii]nvalid"):
            engine_float64.encode(data, num_qubits, encoding_method="iqp_invalid")

    def test_nan_input_rejected(self, engine_float64):
        """Test that NaN input is rejected."""
        data = [float("nan"), 1.0]
        num_qubits = 2

        with pytest.raises((RuntimeError, ValueError)):
            engine_float64.encode(data, num_qubits, encoding_method="iqp")

    def test_inf_input_rejected(self, engine_float64):
        """Test that Inf input is rejected."""
        data = [float("inf"), 1.0]
        num_qubits = 2

        with pytest.raises((RuntimeError, ValueError)):
            engine_float64.encode(data, num_qubits, encoding_method="iqp")
