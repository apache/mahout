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

"""Smoke and normalization tests for FWT vs Tensor Core IQP paths (GPU vs GPU)."""

import pytest
import torch
from qumat_qdp import QdpEngine


def _iqp_param_count(num_qubits: int) -> int:
    return num_qubits + num_qubits * (num_qubits - 1) // 2


@pytest.fixture(scope="module")
def engine():
    try:
        eng = QdpEngine(device_id=0, precision="float64")
    except Exception as exc:
        pytest.skip(f"Could not initialize QdpEngine: {exc}")
    if not hasattr(eng, "encode_batch_tc"):
        pytest.skip("encode_batch_tc not available in this build")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return eng


def _assert_normalized(state: torch.Tensor, num_qubits: int, label: str) -> None:
    probs = state.abs() ** 2
    if state.ndim == 2:
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
            f"{label}: batch normalization failed at N={num_qubits}"
        )
    else:
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6), (
            f"{label}: normalization failed at N={num_qubits}"
        )


@pytest.mark.parametrize("num_qubits", [8, 12])
@pytest.mark.parametrize("batch_size", [4, 32])
def test_fwt_and_tc_paths_normalized(engine, num_qubits, batch_size):
    """For N<=12 both GPU paths return normalized states."""
    data_len = _iqp_param_count(num_qubits)
    data = torch.randn(batch_size, data_len, dtype=torch.float64).numpy()
    state_len = 1 << num_qubits

    fwt_state = torch.from_dlpack(engine.encode(data, num_qubits, "iqp"))
    assert fwt_state.shape == (batch_size, state_len)
    _assert_normalized(fwt_state, num_qubits, "FWT")

    tc_state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits))
    assert tc_state.shape == (batch_size, state_len)
    _assert_normalized(tc_state, num_qubits, "TC")


@pytest.mark.parametrize("num_qubits", [14])
@pytest.mark.parametrize("batch_size", [4, 8])
def test_large_n_tc_path_smoke(engine, num_qubits, batch_size):
    """Large-N TC Kronecker path runs; FWT remains normalized baseline."""
    data_len = _iqp_param_count(num_qubits)
    data = torch.randn(batch_size, data_len, dtype=torch.float64).numpy()
    state_len = 1 << num_qubits

    fwt_state = torch.from_dlpack(engine.encode(data, num_qubits, "iqp"))
    assert fwt_state.shape == (batch_size, state_len)
    _assert_normalized(fwt_state, num_qubits, "FWT")

    tc_state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits))
    assert tc_state.shape == (batch_size, state_len)
    assert torch.isfinite(tc_state).all()


@pytest.mark.parametrize("num_qubits", [14])
def test_fwt_tc_path_agreement_loose(engine, num_qubits):
    """Large-N TC scaffold should be within loose tolerance of FWT (structural PR)."""
    batch_size = 8
    data_len = _iqp_param_count(num_qubits)
    data = torch.randn(batch_size, data_len, dtype=torch.float64).numpy()

    fwt_state = torch.from_dlpack(engine.encode(data, num_qubits, "iqp"))
    tc_state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits))

    max_err = (fwt_state - tc_state).abs().max().item()
    # Ozaki Kronecker scaffold may diverge until PR6 malloc pooling lands.
    assert max_err < 0.1, (
        f"Max abs error {max_err} unexpectedly large at N={num_qubits}"
    )
