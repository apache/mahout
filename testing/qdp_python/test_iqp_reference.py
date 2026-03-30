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

"""IQP numerical correctness tests against the torch reference."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

# Allow importing benchmark helpers from qdp/qdp-python/benchmark.
_sys = __import__("sys")
_qdp_python = Path(__file__).resolve().parent.parent.parent / "qdp" / "qdp-python"
_bench_dir = _qdp_python / "benchmark"
if _qdp_python.exists() and str(_qdp_python) not in _sys.path:
    _sys.path.insert(0, str(_qdp_python))
if _bench_dir.exists() and str(_bench_dir) not in _sys.path:
    _sys.path.insert(0, str(_bench_dir))

torch = pytest.importorskip("torch")
from benchmark.iqp_reference import (
    build_iqp_reference,
    iqp_full_data_len,
    iqp_reference_torch,
    iqp_sample_size,
)

from testing.qdp.qdp_test_utils import requires_qdp


def _engine_float64():
    from _qdp import QdpEngine

    return QdpEngine(0, precision="float64")


def _assert_state_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape, (
        f"shape mismatch: {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"dtype mismatch: {actual.dtype} vs {expected.dtype}"
    )
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10), (
        f"state mismatch\nactual={actual}\nexpected={expected}"
    )


@requires_qdp
@pytest.mark.gpu
def test_iqp_reference_single_qubit_exact_phase_flip() -> None:
    """A single pi phase should map |+> to |1> with high precision."""
    engine = _engine_float64()
    data = [math.pi]
    qtensor = engine.encode(data, 1, "iqp")
    actual = torch.from_dlpack(qtensor)
    expected = iqp_reference_torch(data, 1, enable_zz=True, device="cuda")
    _assert_state_close(actual, expected)
    reference = torch.tensor([[0.0, 1.0]], dtype=torch.complex128, device="cuda")
    _assert_state_close(actual, reference)


@requires_qdp
@pytest.mark.gpu
def test_iqp_reference_matches_hard_full_cases() -> None:
    """Compare the CUDA kernel with the torch reference on harder full-IQP inputs."""
    engine = _engine_float64()
    cases = [
        (2, [0.0, math.pi, -math.pi / 2.0]),
        (
            4,
            [
                0.0,
                math.pi,
                -math.pi / 2.0,
                math.pi / 3.0,
                -2.0 * math.pi,
                math.pi / 7.0,
                -math.pi,
                0.25,
                -0.75,
                1.5,
            ],
        ),
        (5, [(idx - 7) * 0.2 * math.pi for idx in range(iqp_full_data_len(5))]),
    ]

    for num_qubits, data in cases:
        qtensor = engine.encode(data, num_qubits, "iqp")
        actual = torch.from_dlpack(qtensor)
        expected = iqp_reference_torch(data, num_qubits, enable_zz=True, device="cuda")
        _assert_state_close(actual, expected)


@requires_qdp
@pytest.mark.gpu
def test_iqp_reference_matches_hard_batch_cases() -> None:
    """Compare batched IQP CUDA output with the torch reference."""
    engine = _engine_float64()
    num_qubits = 4
    sample_size = iqp_sample_size(num_qubits, enable_zz=True)
    batch = [
        [(idx - 3) * 0.1 * math.pi for idx in range(sample_size)],
        [math.sin(idx) * 0.25 * math.pi for idx in range(sample_size)],
        [(sample_size - idx - 1) * 0.125 * math.pi for idx in range(sample_size)],
    ]

    qtensor = engine.encode(
        torch.tensor(batch, dtype=torch.float64, device="cuda"), num_qubits, "iqp"
    )
    actual = torch.from_dlpack(qtensor)
    expected = build_iqp_reference(
        num_qubits,
        enable_zz=True,
        device="cuda",
        dtype=torch.float64,
    )(torch.tensor(batch, dtype=torch.float64, device="cuda"))
    _assert_state_close(actual, expected)


@requires_qdp
@pytest.mark.gpu
def test_iqp_z_reference_matches_hard_cases() -> None:
    """IQP-Z should follow the same torch reference with ZZ terms disabled."""
    engine = _engine_float64()
    num_qubits = 6
    data = [
        -math.pi,
        -math.pi / 2.0,
        -math.pi / 3.0,
        math.pi / 5.0,
        math.pi / 7.0,
        0.0,
    ]

    qtensor = engine.encode(data, num_qubits, "iqp-z")
    actual = torch.from_dlpack(qtensor)
    expected = iqp_reference_torch(data, num_qubits, enable_zz=False, device="cuda")
    _assert_state_close(actual, expected)
