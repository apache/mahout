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

import numpy as np
import pytest
import torch
from qumat_qdp import QdpEngine

TOLERANCES = {
    # N <= 12: shared-mem fusion has full FP64 precision
    "small_n": {"atol": 1e-9, "rtol": 1e-9},
    # N > 12: Ozaki 7-prime INT8 TC CRT accumulation error
    "large_n": {"atol": 1e-5, "rtol": 1e-5},
}


def _iqp_param_count(num_qubits: int) -> int:
    return num_qubits + num_qubits * (num_qubits - 1) // 2


def torch_iqp_encode_ref(
    data: np.ndarray,
    num_qubits: int,
    enable_zz: bool = True,
) -> torch.Tensor:
    """CPU formula oracle — independent of GPU kernels."""
    batch_size = data.shape[0]
    state_len = 1 << num_qubits
    states = torch.zeros(batch_size, state_len, dtype=torch.complex128)

    for s in range(batch_size):
        params = data[s]
        for x in range(state_len):
            phase = sum(params[i] * float((x >> i) & 1) for i in range(num_qubits))
            if enable_zz:
                idx = num_qubits
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        phase += params[idx] * float(((x >> i) & 1) & ((x >> j) & 1))
                        idx += 1
            states[s, x] = torch.exp(torch.tensor(1j * phase, dtype=torch.complex128))

    norms = states.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
    states = states / norms

    # Apply final Hadamard (FWT) to match IQP circuit H D H |0>
    for stage in range(num_qubits):
        stride = 1 << stage
        reshaped = states.view(batch_size, -1, 2, stride)
        a = reshaped[:, :, 0, :].clone()
        b = reshaped[:, :, 1, :].clone()
        reshaped[:, :, 0, :] = a + b
        reshaped[:, :, 1, :] = a - b

    return states / (2.0 ** (num_qubits / 2.0))


@pytest.fixture(params=[42, 137], ids=["seed42", "seed137"])
def rng_seed(request):
    """Two fixed seeds to catch seed-specific precision cliffs."""
    return request.param


def make_data(
    batch_size: int,
    num_qubits: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    """Deterministic: (batch_size-2) general + zero-phase row + pi/2 row."""
    torch.manual_seed(seed)
    n_params = _iqp_param_count(num_qubits)
    general = torch.randn(batch_size - 2, n_params, dtype=dtype)
    zero_row = torch.zeros(1, n_params, dtype=dtype)
    pi_half = torch.full((1, n_params), torch.pi / 2, dtype=dtype)
    return torch.cat([general, zero_row, pi_half], dim=0).numpy()


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


@pytest.mark.parametrize("num_qubits", [6, 8, 10, 12])
@pytest.mark.parametrize("batch_size", [4])
def test_tc_vs_formula_ref_small_n(engine, num_qubits, batch_size, rng_seed):
    """
    Small-N shared-mem fusion path must match the closed-form formula oracle.
    Tolerance: 1e-9 (FP64 shared-memory kernel — no Ozaki approximation).
    """
    data = make_data(batch_size, num_qubits, seed=rng_seed)
    tc_state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits)).clone().cpu()
    ref_state = torch_iqp_encode_ref(data, num_qubits)
    # Compare probabilities (gauge-invariant; phase rotation is unobservable)
    tc_probs = tc_state.abs().pow(2).double()
    ref_probs = ref_state.abs().pow(2)
    max_err = (tc_probs - ref_probs).abs().max().item()
    assert max_err < TOLERANCES["small_n"]["atol"], (
        f"TC vs formula mismatch (N={num_qubits}): max_err={max_err:.2e}"
    )


@pytest.mark.parametrize("num_qubits", [14, 16])
def test_tc_vs_formula_ref_large_n(engine, num_qubits, rng_seed):
    """
    Large-N Kronecker + Ozaki CRT path: verify against formula reference.
    Tolerance: 1e-5, consistent with 7-prime INT8 TC accumulation error bound.
    """
    batch_size = 4  # keep reference computation feasible on CI
    data = make_data(batch_size, num_qubits, seed=rng_seed)
    tc_state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits)).clone().cpu()
    ref_state = torch_iqp_encode_ref(data, num_qubits)
    tc_probs = tc_state.abs().pow(2).double()
    ref_probs = ref_state.abs().pow(2)
    max_err = (tc_probs - ref_probs).abs().max().item()
    assert max_err < TOLERANCES["large_n"]["atol"], (
        f"Ozaki CRT error {max_err:.2e} exceeds bound at N={num_qubits}"
    )


def test_normalization_zero_phase(engine):
    """
    All-zero phase params -> delta function at |0>.
    The intermediate state is uniform superposition, which is the hardest case for Ozaki CRT (all-one matrix),
    and the final FWT transforms it to a delta function.
    """
    for num_qubits in [8, 14]:
        data = np.zeros((1, _iqp_param_count(num_qubits)), dtype=np.float64)
        state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits)).cpu()
        expected = torch.zeros_like(state.abs())
        expected[:, 0] = 1.0
        assert torch.allclose(
            state.abs(),
            expected,
            atol=1e-9,
        ), f"Delta state failed at N={num_qubits}"


@pytest.mark.parametrize("num_qubits", [8, 12])
@pytest.mark.parametrize("batch_size", [4, 32])
def test_fwt_and_tc_paths_normalized(engine, num_qubits, batch_size):
    """For N<=12 both GPU paths return normalized states."""
    data = make_data(batch_size, num_qubits, seed=42)
    state_len = 1 << num_qubits

    fwt_state = torch.from_dlpack(engine.encode(data, num_qubits, "iqp")).cpu()
    assert fwt_state.shape == (batch_size, state_len)
    _assert_normalized(fwt_state, num_qubits, "FWT")

    tc_state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits))
    assert tc_state.shape == (batch_size, state_len)
    _assert_normalized(tc_state, num_qubits, "TC")


@pytest.mark.parametrize("num_qubits", [14, 16])
@pytest.mark.parametrize("batch_size", [4, 8])
def test_large_n_tc_path_smoke(engine, num_qubits, batch_size):
    """Large-N TC Kronecker path runs; FWT remains normalized baseline."""
    data = make_data(batch_size, num_qubits, seed=42)
    state_len = 1 << num_qubits

    fwt_state = torch.from_dlpack(engine.encode(data, num_qubits, "iqp")).cpu()
    assert fwt_state.shape == (batch_size, state_len)
    _assert_normalized(fwt_state, num_qubits, "FWT")

    tc_state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits))
    assert tc_state.shape == (batch_size, state_len)
    assert torch.isfinite(tc_state).all()


@pytest.mark.parametrize("num_qubits", [14, 16, 17, 18])
def test_fwt_tc_path_agreement_loose(engine, num_qubits):
    """Large-N Kronecker TC path should match FWT within Ozaki tolerance."""
    batch_size = 8
    data = make_data(batch_size, num_qubits, seed=42)

    fwt_state = torch.from_dlpack(engine.encode(data, num_qubits, "iqp")).cpu().clone().cpu()
    tc_state = torch.from_dlpack(engine.encode_batch_tc(data, num_qubits)).clone().cpu().cpu()

    max_err = (fwt_state - tc_state).abs().max().item()
    # Ensure error is within the Ozaki 7-prime CRT bound (TOLERANCES["large_n"]["atol"])
    assert max_err < 1e-5, (
        f"Max abs error {max_err} unexpectedly large at N={num_qubits}"
    )
