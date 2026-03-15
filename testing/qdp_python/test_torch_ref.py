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
Tests for pure-PyTorch reference encoding implementations.

These tests run on CPU and do NOT require the _qdp Rust extension.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from qumat_qdp.torch_ref import (
    amplitude_encode,
    angle_encode,
    basis_encode,
    encode,
    iqp_encode,
)

# ---------------------------------------------------------------------------
# Amplitude encoding
# ---------------------------------------------------------------------------


class TestAmplitudeEncode:
    def test_normalization(self):
        data = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=2)
        expected = torch.tensor(
            [[0.6 + 0j, 0.8 + 0j, 0.0 + 0j, 0.0 + 0j]], dtype=torch.complex128
        )
        assert torch.allclose(result, expected, atol=1e-10)

    def test_padding(self):
        data = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=2)
        assert result.shape == (1, 4)
        # After padding [1, 0, 0, 0] and normalizing: [1, 0, 0, 0]
        assert torch.allclose(
            result[0, 0], torch.tensor(1.0 + 0j, dtype=torch.complex128)
        )
        assert torch.allclose(result[0, 1:], torch.zeros(3, dtype=torch.complex128))

    def test_batch(self):
        data = torch.randn(5, 8, dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=3)
        assert result.shape == (5, 8)

    def test_1d_input(self):
        data = torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=2)
        assert result.shape == (1, 4)

    def test_zero_vector_near_zero(self):
        """Zero-norm row produces near-zero output (no GPU sync error)."""
        data = torch.zeros(1, 4, dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=2)
        assert torch.allclose(result.abs(), torch.zeros_like(result.abs()), atol=1e-5)

    def test_nan_propagates(self):
        """NaN input propagates through output (no GPU sync validation)."""
        data = torch.tensor([[float("nan"), 1.0, 0.0, 0.0]], dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=2)
        assert torch.any(torch.isnan(result.real))

    def test_features_exceed_state_dim_raises(self):
        data = torch.randn(1, 16, dtype=torch.float64)
        with pytest.raises(ValueError, match="exceed state dimension"):
            amplitude_encode(data, num_qubits=2)  # state_dim=4

    def test_unit_norm(self):
        data = torch.randn(10, 8, dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=3)
        norms = torch.abs(result).norm(dim=1)
        assert torch.allclose(norms, torch.ones(10, dtype=torch.float64), atol=1e-10)

    def test_dtype_float32(self):
        data = torch.randn(2, 4, dtype=torch.float32)
        result = amplitude_encode(data, num_qubits=2)
        assert result.dtype == torch.complex64

    def test_dtype_float64(self):
        data = torch.randn(2, 4, dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=2)
        assert result.dtype == torch.complex128


# ---------------------------------------------------------------------------
# Angle encoding
# ---------------------------------------------------------------------------


class TestAngleEncode:
    def test_zero_angles(self):
        """[0, 0] -> |00> = [1, 0, 0, 0]."""
        data = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        result = angle_encode(data, num_qubits=2)
        expected = torch.tensor(
            [[1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j]], dtype=torch.complex128
        )
        assert torch.allclose(result, expected, atol=1e-10)

    def test_pi_half_first_qubit(self):
        """[pi/2, 0] -> cos(pi/2)*cos(0)=0, sin(pi/2)*cos(0)=1, ..."""
        data = torch.tensor([[math.pi / 2, 0.0]], dtype=torch.float64)
        result = angle_encode(data, num_qubits=2)
        # State |01>: bit0=1 -> sin(pi/2)=1, bit1=0 -> cos(0)=1 => amplitude=1
        assert abs(result[0, 0].real) < 1e-10  # |00>
        assert abs(result[0, 1].real - 1.0) < 1e-10  # |01>
        assert abs(result[0, 2].real) < 1e-10  # |10>
        assert abs(result[0, 3].real) < 1e-10  # |11>

    def test_wrong_length_raises(self):
        data = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        with pytest.raises(RuntimeError, match="expects 2 values"):
            angle_encode(data, num_qubits=2)

    def test_unit_norm(self):
        data = torch.randn(10, 4, dtype=torch.float64)
        result = angle_encode(data, num_qubits=4)
        norms = torch.abs(result).norm(dim=1)
        assert torch.allclose(norms, torch.ones(10, dtype=torch.float64), atol=1e-10)

    def test_batch_shape(self):
        data = torch.randn(7, 3, dtype=torch.float64)
        result = angle_encode(data, num_qubits=3)
        assert result.shape == (7, 8)

    def test_1d_input(self):
        data = torch.tensor([0.5, 1.0], dtype=torch.float64)
        result = angle_encode(data, num_qubits=2)
        assert result.shape == (1, 4)


# ---------------------------------------------------------------------------
# Basis encoding
# ---------------------------------------------------------------------------


class TestBasisEncode:
    def test_index_zero(self):
        data = torch.tensor([0.0], dtype=torch.float64)
        result = basis_encode(data, num_qubits=2)
        expected = torch.tensor([[1.0 + 0j, 0, 0, 0]], dtype=torch.complex128)
        assert torch.allclose(result, expected)

    def test_index_three(self):
        data = torch.tensor([3.0], dtype=torch.float64)
        result = basis_encode(data, num_qubits=2)
        assert result[0, 3].real == 1.0
        assert result[0, :3].abs().sum() == 0.0

    def test_batch(self):
        data = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        result = basis_encode(data, num_qubits=2)
        assert result.shape == (4, 4)
        for i in range(4):
            assert result[i, i].real == 1.0

    def test_2d_input(self):
        data = torch.tensor([[2.0]])
        result = basis_encode(data, num_qubits=2)
        assert result.shape == (1, 4)
        assert result[0, 2].real == 1.0

    def test_out_of_range_raises(self):
        data = torch.tensor([4.0])
        with pytest.raises(RuntimeError, match="exceeds state vector size"):
            basis_encode(data, num_qubits=2)

    def test_negative_raises(self):
        data = torch.tensor([-1.0])
        with pytest.raises(RuntimeError, match="non-negative"):
            basis_encode(data, num_qubits=2)

    def test_non_integer_raises(self):
        data = torch.tensor([1.5])
        with pytest.raises(RuntimeError, match="integer-valued"):
            basis_encode(data, num_qubits=2)


# ---------------------------------------------------------------------------
# IQP encoding
# ---------------------------------------------------------------------------


class TestIqpEncode:
    def test_zero_params_gives_zero_state(self):
        """All-zero params → H^n I H^n |0⟩ = |0⟩ = [1, 0, ..., 0]."""
        n = 3
        data = torch.zeros(1, n, dtype=torch.float64)
        result = iqp_encode(data, num_qubits=n, enable_zz=False)
        expected = torch.zeros(1, 1 << n, dtype=torch.complex128)
        expected[0, 0] = 1.0 + 0j
        assert torch.allclose(result, expected, atol=1e-10)

    def test_z_only_mode(self):
        n = 3
        data = torch.randn(2, n, dtype=torch.float64)
        result = iqp_encode(data, num_qubits=n, enable_zz=False)
        assert result.shape == (2, 1 << n)

    def test_zz_mode(self):
        n = 3
        n_params = n + n * (n - 1) // 2  # 3 + 3 = 6
        data = torch.randn(2, n_params, dtype=torch.float64)
        result = iqp_encode(data, num_qubits=n, enable_zz=True)
        assert result.shape == (2, 1 << n)

    def test_unit_norm(self):
        n = 4
        n_params = n + n * (n - 1) // 2
        data = torch.randn(5, n_params, dtype=torch.float64)
        result = iqp_encode(data, num_qubits=n, enable_zz=True)
        norms = torch.abs(result).norm(dim=1)
        assert torch.allclose(norms, torch.ones(5, dtype=torch.float64), atol=1e-10)

    def test_wrong_param_count_raises(self):
        with pytest.raises(RuntimeError, match="expects"):
            iqp_encode(
                torch.randn(1, 5, dtype=torch.float64), num_qubits=3, enable_zz=False
            )

    def test_1d_input(self):
        data = torch.zeros(3, dtype=torch.float64)
        result = iqp_encode(data, num_qubits=3, enable_zz=False)
        assert result.shape == (1, 8)

    def test_default_enable_zz(self):
        """Default enable_zz=True."""
        n = 2
        n_params = n + n * (n - 1) // 2  # 2 + 1 = 3
        data = torch.randn(1, n_params, dtype=torch.float64)
        result = iqp_encode(data, num_qubits=n)
        assert result.shape == (1, 4)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_amplitude(self):
        data = torch.randn(2, 4, dtype=torch.float64)
        result = encode(data, num_qubits=2, encoding_method="amplitude")
        assert result.shape == (2, 4)

    def test_angle(self):
        data = torch.randn(2, 3, dtype=torch.float64)
        result = encode(data, num_qubits=3, encoding_method="angle")
        assert result.shape == (2, 8)

    def test_basis(self):
        data = torch.tensor([0.0, 1.0])
        result = encode(data, num_qubits=2, encoding_method="basis")
        assert result.shape == (2, 4)

    def test_iqp(self):
        data = torch.randn(1, 3, dtype=torch.float64)
        result = encode(data, num_qubits=3, encoding_method="iqp", enable_zz=False)
        assert result.shape == (1, 8)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown encoding method"):
            encode(torch.randn(1, 4), num_qubits=2, encoding_method="invalid")


# ---------------------------------------------------------------------------
# Device placement (CPU always; GPU if available)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_integer_dtype_raises(self):
        data = torch.tensor([[1, 2, 3, 4]])
        with pytest.raises(ValueError, match="floating-point"):
            amplitude_encode(data, num_qubits=2)

    def test_3d_tensor_raises(self):
        data = torch.randn(2, 3, 4, dtype=torch.float64)
        with pytest.raises(ValueError, match="2-D"):
            amplitude_encode(data, num_qubits=2)

    def test_angle_nan_propagates(self):
        """NaN propagates through angle encoding."""
        data = torch.tensor([[float("nan"), 0.0]], dtype=torch.float64)
        result = angle_encode(data, num_qubits=2)
        assert torch.any(torch.isnan(result.real))

    def test_iqp_inf_propagates(self):
        """Inf propagates through IQP encoding."""
        data = torch.tensor([[float("inf"), 0.0, 0.0]], dtype=torch.float64)
        result = iqp_encode(data, num_qubits=3, enable_zz=False)
        # Inf in phase → cos/sin produce NaN
        assert torch.any(torch.isnan(result.real))

    def test_basis_multi_column_raises(self):
        data = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        with pytest.raises(ValueError, match="\\(batch,\\) or \\(batch, 1\\)"):
            basis_encode(data, num_qubits=2)

    def test_iqp_zz_wrong_param_count_raises(self):
        with pytest.raises(RuntimeError, match="expects"):
            iqp_encode(
                torch.randn(1, 3, dtype=torch.float64), num_qubits=3, enable_zz=True
            )

    def test_basis_integer_tensor(self):
        """basis_encode accepts integer tensors directly."""
        data = torch.tensor([0, 3], dtype=torch.long)
        result = basis_encode(data, num_qubits=2)
        assert result.dtype == torch.complex128
        assert result[0, 0].real == 1.0
        assert result[1, 3].real == 1.0

    def test_unsupported_dtype_raises(self):
        data = torch.randn(2, 4).half()  # float16
        with pytest.raises(TypeError, match="Unsupported dtype"):
            amplitude_encode(data, num_qubits=2)


class TestDevicePlacement:
    def test_cpu_output(self):
        data = torch.randn(2, 4, dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=2, device="cpu")
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_output(self):
        data = torch.randn(2, 4, dtype=torch.float64)
        result = amplitude_encode(data, num_qubits=2, device="cuda:0")
        assert result.device.type == "cuda"


# ---------------------------------------------------------------------------
# Cross-validation: torch_ref vs _qdp (only runs when Rust extension is available)
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """Compare torch_ref output against _qdp output for the same inputs."""

    @pytest.fixture(autouse=True)
    def _require_qdp(self):
        pytest.importorskip("_qdp")

    @pytest.mark.gpu
    @pytest.mark.parametrize("encoding", ["amplitude", "angle", "basis", "iqp"])
    def test_encoding_matches_rust(self, encoding):
        import _qdp
        import numpy as np

        engine = _qdp.QdpEngine(0)
        num_qubits = 3
        state_dim = 1 << num_qubits

        if encoding == "basis":
            np_data = np.array([[0.0], [3.0], [7.0]])
        elif encoding == "angle":
            np_data = np.random.rand(4, num_qubits).astype(np.float64) * 2 * 3.14159
        elif encoding == "iqp":
            n_params = num_qubits + num_qubits * (num_qubits - 1) // 2
            np_data = np.random.rand(4, n_params).astype(np.float64)
        else:
            np_data = np.random.rand(4, state_dim).astype(np.float64)

        # Rust path
        rust_qt = engine.encode(
            np_data, num_qubits=num_qubits, encoding_method=encoding
        )
        rust_tensor = torch.from_dlpack(rust_qt)

        # PyTorch reference path
        pt_data = torch.tensor(np_data, dtype=torch.float64, device="cuda:0")
        if encoding == "basis":
            pt_data = pt_data.flatten()
        ref_tensor = encode(pt_data, num_qubits, encoding, device="cuda:0")

        assert torch.allclose(
            rust_tensor.to(torch.complex128),
            ref_tensor.to(torch.complex128),
            atol=1e-10,
        ), f"Mismatch for {encoding} encoding"
