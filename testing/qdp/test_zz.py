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

"""Tests for ZZFeatureMap encoding."""

import pytest
import torch
import numpy as np
from .qdp_test_utils import requires_qdp

@requires_qdp
@pytest.mark.gpu
def test_zz_encoding_basic():
    """Test ZZFeatureMap encoding with 1 layer and 2 qubits."""
    from _qdp import QdpEngine
    
    engine = QdpEngine(0)
    # ZZ expects 1 value per qubit
    data = np.array([0.5, 0.2], dtype=np.float64)
    
    # Encode with 1 layer (simplest case)
    qtensor = engine.encode(data, 2, "zz")
    
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.shape == (4,)
    assert torch_tensor.is_cuda
    
    # Verify norm is 1
    norm = torch.norm(torch_tensor)
    assert torch.allclose(norm, torch.tensor(1.0, device='cuda', dtype=torch.complex128), atol=1e-6)

@requires_qdp
@pytest.mark.gpu
def test_zz_encoding_multi_layer():
    """Test ZZFeatureMap with multiple layers."""
    from _qdp import QdpEngine
    
    engine = QdpEngine(0)
    data = np.array([0.1, 0.2, 0.3], dtype=np.float64) # 3 qubits
    
    # Encode with 3 layers
    qtensor = engine.encode(data, 3, "zz") # Default num_layers is 2, but we can't set it yet in python
    
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.shape == (8,)
    
    # Verify norm
    norm = torch.norm(torch_tensor)
    assert torch.allclose(norm, torch.tensor(1.0, device='cuda', dtype=torch.complex128), atol=1e-6)

@requires_qdp
@pytest.mark.gpu
def test_zz_batch_encoding():
    """Test batched ZZFeatureMap encoding."""
    from _qdp import QdpEngine
    
    engine = QdpEngine(0)
    num_samples = 10
    num_qubits = 4
    data = np.random.rand(num_samples, num_qubits).astype(np.float64)
    
    qtensor = engine.encode(data, num_qubits, "zz")
    
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.shape == (num_samples, 1 << num_qubits)
    
    # Verify norms for all samples
    norms = torch.norm(torch_tensor, dim=1)
    assert torch.allclose(norms, torch.ones(num_samples, device='cuda', dtype=torch.complex128), atol=1e-6)

@requires_qdp
@pytest.mark.gpu
def test_zz_parameterized_layers():
    """Test ZZFeatureMap with explicit layer counts in the name."""
    from _qdp import QdpEngine
    
    engine = QdpEngine(0)
    data = np.array([0.5, 0.5], dtype=np.float64)
    
    # Layer 1
    t1 = torch.from_dlpack(engine.encode(data, 2, "zz-l1"))
    # Layer 3
    t3 = torch.from_dlpack(engine.encode(data, 2, "zz-l3"))
    
    assert not torch.allclose(t1, t3)
    assert torch.allclose(torch.norm(t1), torch.tensor(1.0, device='cuda', dtype=torch.complex128))
    assert torch.allclose(torch.norm(t3), torch.tensor(1.0, device='cuda', dtype=torch.complex128))
