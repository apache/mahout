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

import pytest
import torch
from qumat_qdp import QdpEngine
from qumat_qdp.torch_ref import iqp_encode as iqp_encode_baseline


@pytest.fixture(scope="module")
def engine():
    try:
        return QdpEngine(precision="float64")
    except Exception as e:
        pytest.skip(f"Could not initialize QdpEngine: {e}")


@pytest.mark.parametrize("n_qubits", [2, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("enable_zz", [True, False])
def test_implicit_fwt_correctness(engine, n_qubits, batch_size, enable_zz):
    """
    Test that the QDP engine's implicit FWT logic perfectly matches
    the theoretical exact PyTorch implementation.
    """
    # Expected number of parameters for IQP
    if enable_zz:
        n_params = n_qubits + n_qubits * (n_qubits - 1) // 2
        method = "iqp"
    else:
        n_params = n_qubits
        method = "iqp-z"

    # Generate random parameters
    data = torch.randn(batch_size, n_params, dtype=torch.float64, device="cuda")

    # 1. Baseline logic (pure PyTorch)
    expected_state = iqp_encode_baseline(
        data, n_qubits, enable_zz=enable_zz, device="cuda"
    )

    # 2. QDP implicit FWT logic
    # The QDP engine internally dispatches to standard SIMT implicit FWT kernel
    actual_state_dlpack = engine.encode(data, n_qubits, encoding_method=method)
    actual_state = torch.from_dlpack(actual_state_dlpack)

    # 3. Validation
    # We use a strict tolerance since both should be FP64 deterministic computations
    torch.testing.assert_close(
        actual_state,
        expected_state,
        rtol=1e-12,
        atol=1e-12,
        msg=f"Mismatch found for N={n_qubits}, enable_zz={enable_zz}",
    )
