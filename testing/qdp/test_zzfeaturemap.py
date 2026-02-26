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

"""Tests for ZZFeatureMap encoding and its QDP implementation."""

import math
import pytest
import numpy as np

from .qdp_test_utils import requires_qdp


def _has_qiskit():
    """Check if Qiskit is available for reference comparisons."""
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.quantum_info import Statevector
        return True
    except ImportError:
        return False


requires_qiskit = pytest.mark.skipif(
    not _has_qiskit(),
    reason="Qiskit not installed. Install with: pip install qiskit"
)


def qiskit_zzfeaturemap_statevector(features, num_qubits, reps=2, entanglement='full'):
    """Return reference statevector from Qiskit's ZZFeatureMap."""
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Statevector
    
    qc = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=reps,
        entanglement=entanglement,
    )
    
    # Bind parameters: Qiskit uses features directly (not pre-expanded angles)
    param_values = {p: features[i % len(features)] for i, p in enumerate(qc.parameters)}
    bound_qc = qc.assign_parameters(param_values)
    
    sv = Statevector.from_instruction(bound_qc)
    return sv.data


def expand_features_to_qdp_params(features, num_qubits, reps, entanglement='full'):
    """Expand raw features to QDP's flat ZZFeatureMap parameter layout."""
    x = np.asarray(features, dtype=np.float64)
    
    # Determine pairs based on entanglement
    if entanglement == 'full':
        pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    elif entanglement == 'linear':
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
    elif entanglement == 'circular':
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        if num_qubits > 1:
            pairs.append((num_qubits - 1, 0))
    else:
        raise ValueError(f"Unknown entanglement: {entanglement}")
    
    params = []
    for layer in range(reps):
        # Single-qubit Z angles: alpha_i = x_i
        for i in range(num_qubits):
            params.append(x[i % len(x)])
        
        # Two-qubit ZZ angles: beta_ij = (pi - x_i) * (pi - x_j)
        for (i, j) in pairs:
            xi = x[i % len(x)]
            xj = x[j % len(x)]
            params.append((math.pi - xi) * (math.pi - xj))
    
    return np.array(params, dtype=np.float64)


def statevectors_equal_up_to_global_phase(sv1, sv2, atol=1e-6):
    """Return True if two statevectors are equal up to global phase."""
    sv1 = np.asarray(sv1, dtype=np.complex128)
    sv2 = np.asarray(sv2, dtype=np.complex128)
    
    if sv1.shape != sv2.shape:
        return False
    
    # Find the first non-negligible amplitude to determine phase
    for i in range(len(sv1)):
        if abs(sv1[i]) > atol and abs(sv2[i]) > atol:
            phase = sv1[i] / sv2[i]
            phase_normalized = phase / abs(phase)  # Unit magnitude
            sv2_aligned = sv2 * phase_normalized
            return np.allclose(sv1, sv2_aligned, atol=atol)
    
    # If we get here, one or both statevectors are near-zero
    return np.allclose(sv1, sv2, atol=atol)


# =============================================================================
# Unit Tests
# =============================================================================

class TestZZFeatureMapUnit:
    """Unit tests for ZZFeatureMap encoder (no GPU required)."""
    
    @requires_qdp
    def test_encoder_available(self):
        """Test that ZZFeatureMap encoder can be loaded by name."""
        from _qdp import QdpEngine
        
        # This test just checks the encoder is registered
        # Actual encoding requires GPU
        engine = QdpEngine(0)
        assert engine is not None
    
    def test_parameter_expansion_full(self):
        """Test parameter expansion for full entanglement."""
        features = [0.5, 0.7]
        num_qubits = 2
        reps = 1
        
        params = expand_features_to_qdp_params(features, num_qubits, reps, 'full')
        
        # Expected: [alpha_0, alpha_1, beta_01]
        # alpha_0 = 0.5, alpha_1 = 0.7
        # beta_01 = (pi - 0.5) * (pi - 0.7)
        expected_beta = (math.pi - 0.5) * (math.pi - 0.7)
        
        assert len(params) == 3  # n + n*(n-1)/2 = 2 + 1 = 3
        assert abs(params[0] - 0.5) < 1e-10
        assert abs(params[1] - 0.7) < 1e-10
        assert abs(params[2] - expected_beta) < 1e-10
    
    def test_parameter_expansion_linear(self):
        """Test parameter expansion for linear entanglement."""
        features = [0.3, 0.5, 0.7]
        num_qubits = 3
        reps = 1
        
        params = expand_features_to_qdp_params(features, num_qubits, reps, 'linear')
        
        # Expected: [alpha_0, alpha_1, alpha_2, beta_01, beta_12]
        # For linear: pairs are (0,1), (1,2)
        assert len(params) == 5  # n + (n-1) = 3 + 2 = 5
    
    def test_parameter_expansion_circular(self):
        """Test parameter expansion for circular entanglement."""
        features = [0.3, 0.5, 0.7]
        num_qubits = 3
        reps = 1
        
        params = expand_features_to_qdp_params(features, num_qubits, reps, 'circular')
        
        # Expected: [alpha_0, alpha_1, alpha_2, beta_01, beta_12, beta_20]
        # For circular: pairs are (0,1), (1,2), (2,0)
        assert len(params) == 6  # n + n = 3 + 3 = 6
    
    def test_parameter_expansion_reps(self):
        """Test parameter expansion with multiple reps."""
        features = [0.5, 0.7]
        num_qubits = 2
        reps = 3
        
        params = expand_features_to_qdp_params(features, num_qubits, reps, 'full')
        
        # Expected: 3 layers × (2 + 1) params = 9 total
        assert len(params) == 9
    
    def test_global_phase_equivalence(self):
        """Test that global phase comparison works correctly."""
        sv1 = np.array([0.5, 0.5, 0.5, 0.5])
        sv2 = sv1 * np.exp(1j * 0.3)  # Same state with global phase
        sv3 = np.array([0.5, -0.5, 0.5, 0.5])  # Different state
        
        assert statevectors_equal_up_to_global_phase(sv1, sv2)
        assert not statevectors_equal_up_to_global_phase(sv1, sv3)

@requires_qdp
@requires_qiskit
@pytest.mark.gpu
class TestZZFeatureMapQiskitValidation:
    """Validate QDP ZZFeatureMap against Qiskit reference (requires GPU + Qiskit)."""
    
    @pytest.mark.parametrize("num_qubits", [2, 3, 4])
    def test_full_entanglement_reps2(self, num_qubits):
        """Test full entanglement with reps=2 (Qiskit default)."""
        from _qdp import QdpEngine
        import torch
        
        engine = QdpEngine(0)
        
        # Use simple feature values
        features = [0.1 * (i + 1) for i in range(num_qubits)]
        
        # Get Qiskit reference
        qiskit_sv = qiskit_zzfeaturemap_statevector(
            features, num_qubits, reps=2, entanglement='full'
        )
        
        # Expand features to QDP parameter format
        qdp_params = expand_features_to_qdp_params(
            features, num_qubits, reps=2, entanglement='full'
        )
        
        # Run QDP encoder
        encoding_method = "zzfeaturemap-full-reps2"
        qtensor = engine.encode(qdp_params.tolist(), num_qubits, encoding_method)
        
        # Convert to numpy for comparison
        torch_tensor = torch.from_dlpack(qtensor)
        qdp_sv = torch_tensor.cpu().numpy().flatten()
        
        assert statevectors_equal_up_to_global_phase(qdp_sv, qiskit_sv), (
            f"QDP and Qiskit statevectors differ for {num_qubits} qubits, full, reps=2"
        )
    
    @pytest.mark.parametrize("num_qubits", [3, 4])
    def test_linear_entanglement(self, num_qubits):
        """Test linear entanglement pattern."""
        from _qdp import QdpEngine
        import torch
        
        engine = QdpEngine(0)
        features = [0.2 * (i + 1) for i in range(num_qubits)]
        
        qiskit_sv = qiskit_zzfeaturemap_statevector(
            features, num_qubits, reps=2, entanglement='linear'
        )
        
        qdp_params = expand_features_to_qdp_params(
            features, num_qubits, reps=2, entanglement='linear'
        )
        
        encoding_method = "zzfeaturemap-linear-reps2"
        qtensor = engine.encode(qdp_params.tolist(), num_qubits, encoding_method)
        
        torch_tensor = torch.from_dlpack(qtensor)
        qdp_sv = torch_tensor.cpu().numpy().flatten()
        
        assert statevectors_equal_up_to_global_phase(qdp_sv, qiskit_sv), (
            f"QDP and Qiskit statevectors differ for {num_qubits} qubits, linear, reps=2"
        )
    
    @pytest.mark.parametrize("num_qubits", [3, 4])
    def test_circular_entanglement(self, num_qubits):
        """Test circular entanglement pattern."""
        from _qdp import QdpEngine
        import torch
        
        engine = QdpEngine(0)
        features = [0.15 * (i + 1) for i in range(num_qubits)]
        
        qiskit_sv = qiskit_zzfeaturemap_statevector(
            features, num_qubits, reps=2, entanglement='circular'
        )
        
        qdp_params = expand_features_to_qdp_params(
            features, num_qubits, reps=2, entanglement='circular'
        )
        
        encoding_method = "zzfeaturemap-circular-reps2"
        qtensor = engine.encode(qdp_params.tolist(), num_qubits, encoding_method)
        
        torch_tensor = torch.from_dlpack(qtensor)
        qdp_sv = torch_tensor.cpu().numpy().flatten()
        
        assert statevectors_equal_up_to_global_phase(qdp_sv, qiskit_sv), (
            f"QDP and Qiskit statevectors differ for {num_qubits} qubits, circular, reps=2"
        )
    
    @pytest.mark.parametrize("reps", [1, 2, 3])
    def test_varying_reps(self, reps):
        """Test different repetition layer counts."""
        from _qdp import QdpEngine
        import torch
        
        engine = QdpEngine(0)
        num_qubits = 2
        features = [0.3, 0.7]
        
        qiskit_sv = qiskit_zzfeaturemap_statevector(
            features, num_qubits, reps=reps, entanglement='full'
        )
        
        qdp_params = expand_features_to_qdp_params(
            features, num_qubits, reps=reps, entanglement='full'
        )
        
        encoding_method = f"zzfeaturemap-full-reps{reps}"
        qtensor = engine.encode(qdp_params.tolist(), num_qubits, encoding_method)
        
        torch_tensor = torch.from_dlpack(qtensor)
        qdp_sv = torch_tensor.cpu().numpy().flatten()
        
        assert statevectors_equal_up_to_global_phase(qdp_sv, qiskit_sv), (
            f"QDP and Qiskit statevectors differ for reps={reps}"
        )


@requires_qdp
@pytest.mark.gpu
class TestZZFeatureMapInputValidation:
    """Test input validation for ZZFeatureMap encoder."""
    
    def test_wrong_data_length(self):
        """Test that wrong data length raises error."""
        from _qdp import QdpEngine
        
        engine = QdpEngine(0)
        
        # For 2 qubits, full, reps=2: expected length = 2 * (2 + 1) = 6
        wrong_data = [0.1] * 5  # Wrong length
        
        with pytest.raises(Exception, match="expects.*values"):
            engine.encode(wrong_data, 2, "zzfeaturemap-full-reps2")
    
    def test_nan_value_rejected(self):
        """Test that NaN values are rejected."""
        from _qdp import QdpEngine
        
        engine = QdpEngine(0)
        
        # For 2 qubits, full, reps=2: expected length = 6
        data_with_nan = [0.1, 0.2, float('nan'), 0.4, 0.5, 0.6]
        
        with pytest.raises(Exception, match="finite"):
            engine.encode(data_with_nan, 2, "zzfeaturemap-full-reps2")
    
    def test_inf_value_rejected(self):
        """Test that infinity values are rejected."""
        from _qdp import QdpEngine
        
        engine = QdpEngine(0)
        
        data_with_inf = [0.1, 0.2, float('inf'), 0.4, 0.5, 0.6]
        
        with pytest.raises(Exception, match="finite"):
            engine.encode(data_with_inf, 2, "zzfeaturemap-full-reps2")
    
    def test_zero_qubits_rejected(self):
        """Test that zero qubits raises error."""
        from _qdp import QdpEngine
        
        engine = QdpEngine(0)
        
        with pytest.raises(Exception, match="at least 1"):
            engine.encode([], 0, "zzfeaturemap")


@requires_qdp
@pytest.mark.gpu
class TestZZFeatureMapNormalization:
    """Test statevector normalization for ZZFeatureMap."""
    
    def test_output_is_normalized(self):
        """Test that output statevector has unit norm."""
        from _qdp import QdpEngine
        import torch
        
        engine = QdpEngine(0)
        
        # 2 qubits, full, reps=2: 6 parameters
        params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        qtensor = engine.encode(params, 2, "zzfeaturemap-full-reps2")
        torch_tensor = torch.from_dlpack(qtensor)
        sv = torch_tensor.cpu().numpy().flatten()
        
        norm = np.sqrt(np.sum(np.abs(sv) ** 2))
        assert abs(norm - 1.0) < 1e-6, f"Statevector norm {norm} != 1.0"
    
    @pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
    def test_output_normalized_various_sizes(self, num_qubits):
        """Test normalization for various qubit counts."""
        from _qdp import QdpEngine
        import torch
        
        engine = QdpEngine(0)
        
        # Calculate expected param length for full entanglement, reps=1
        num_pairs = num_qubits * (num_qubits - 1) // 2
        param_len = 1 * (num_qubits + num_pairs)  # reps=1
        params = [0.1 * i for i in range(param_len)]
        
        qtensor = engine.encode(params, num_qubits, "zzfeaturemap-full-reps1")
        torch_tensor = torch.from_dlpack(qtensor)
        sv = torch_tensor.cpu().numpy().flatten()
        
        norm = np.sqrt(np.sum(np.abs(sv) ** 2))
        assert abs(norm - 1.0) < 1e-6, f"Statevector norm {norm} != 1.0 for {num_qubits} qubits"
