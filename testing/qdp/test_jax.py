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

"""Tests for Jax integration."""

import pytest
import torch
from .qdp_test_utils import requires_qdp

@requires_qdp
@pytest.mark.gpu
def test_jax_integration():
    """Test encoding from Jax array via DLPack."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from _qdp import QdpEngine
    
    engine = QdpEngine(0)
    
    # Create a Jax array on GPU if available, else CPU
    data = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]], dtype=jnp.float64)
    
    # Encode
    qtensor = engine.encode(data, 2, "amplitude")
    
    # Convert to PyTorch to verify
    torch_tensor = torch.from_dlpack(qtensor)
    assert torch_tensor.is_cuda
    assert torch_tensor.shape == (2, 4)
    
    # Verify norms
    norms = torch.sqrt(torch.sum(torch.abs(torch_tensor)**2, dim=1))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

@requires_qdp
@pytest.mark.gpu
def test_jax_block_until_ready():
    """Test that we trigger block_until_ready (via mock if needed, or just by calling)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from _qdp import QdpEngine
    
    engine = QdpEngine(0)
    data = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    
    # This should trigger synchronize_jax_array internally
    qtensor = engine.encode(data, 2, "amplitude")
    assert qtensor is not None
