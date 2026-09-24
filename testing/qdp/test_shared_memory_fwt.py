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


@pytest.mark.parametrize("n_qubits", [8, 10, 12])
@pytest.mark.parametrize("batch_size", [4, 16, 32])
@pytest.mark.parametrize("enable_zz", [True, False])
def test_shared_memory_fwt_batch_correctness(engine, n_qubits, batch_size, enable_zz):
    """Validate fused shared-memory batch IQP encoding for N <= 12."""
    if enable_zz:
        n_params = n_qubits + n_qubits * (n_qubits - 1) // 2
        method = "iqp"
    else:
        n_params = n_qubits
        method = "iqp-z"

    data = torch.randn(batch_size, n_params, dtype=torch.float64, device="cuda")

    expected_state = iqp_encode_baseline(
        data, n_qubits, enable_zz=enable_zz, device="cuda"
    )

    actual_state_dlpack = engine.encode(data, n_qubits, encoding_method=method)
    actual_state = torch.from_dlpack(actual_state_dlpack)

    torch.testing.assert_close(
        actual_state,
        expected_state,
        rtol=1e-12,
        atol=1e-12,
        msg=f"Shared-memory batch mismatch for N={n_qubits}, batch={batch_size}",
    )
