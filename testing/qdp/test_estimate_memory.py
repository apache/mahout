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

"""Tests for the Python memory-estimation API (issue #1430).

Every case here is pure arithmetic on the Rust side: no device is opened and nothing
is allocated, so these run identically on a GPU host and on a stub CUDA build with no
GPU at all.  That is the point of the API -- it exists to size a configuration on a
machine that could not run it.
"""

from __future__ import annotations

import pytest

from .qdp_test_utils import requires_qdp

pytestmark = requires_qdp

MIB = 1024 * 1024


@pytest.fixture
def estimate_memory():
    """The public ``qumat_qdp`` wrapper, which is what users are told to import."""
    from qumat_qdp import estimate_memory as fn

    return fn


def test_known_configuration_matches_the_documented_formulas(estimate_memory):
    """16 qubits, batch 64, f32, depth 16 -- the worked example in the Rust docs.

    Pinning one exact configuration catches a change of formula, of unit, or of field
    order in the binding's return tuple, none of which a relative assertion would.
    """
    est = estimate_memory(
        num_qubits=16, batch_size=64, encoding_method="amplitude", dtype="f32"
    )

    # 16 * 64 * 2**16 samples * 4 bytes
    assert est.cpu_prefetch_bytes == 256 * MIB
    # 2 concurrent buffers * 64 * 2**16 amplitudes * 8 bytes (complex64)
    assert est.gpu_state_bytes == 64 * MIB
    assert est.total_bytes == 320 * MIB


def test_f64_doubles_every_field(estimate_memory):
    """Precision is the knob most likely to be mis-wired: f64 is twice f32 throughout."""
    f32 = estimate_memory(num_qubits=12, batch_size=16, dtype="f32")
    f64 = estimate_memory(num_qubits=12, batch_size=16, dtype="f64")

    assert f64.cpu_prefetch_bytes == 2 * f32.cpu_prefetch_bytes
    assert f64.gpu_state_bytes == 2 * f32.gpu_state_bytes
    assert f64.total_bytes == 2 * f32.total_bytes


def test_prefetch_depth_moves_host_memory_only(estimate_memory):
    """The guard's remedy names batch size and qubits, never prefetch_depth.

    That advice is only correct if device memory really is independent of depth.
    """
    shallow = estimate_memory(num_qubits=10, batch_size=8, prefetch_depth=1)
    deep = estimate_memory(num_qubits=10, batch_size=8, prefetch_depth=8)

    assert deep.cpu_prefetch_bytes == 8 * shallow.cpu_prefetch_bytes
    assert deep.gpu_state_bytes == shallow.gpu_state_bytes


def test_batch_size_scales_device_memory(estimate_memory):
    """Halving batch size halves the device budget -- the remedy the guard suggests."""
    big = estimate_memory(num_qubits=10, batch_size=64)
    small = estimate_memory(num_qubits=10, batch_size=32)

    assert big.gpu_state_bytes == 2 * small.gpu_state_bytes


def test_device_state_is_2_to_the_n_whatever_the_input_width(estimate_memory):
    """Angle input is n values per sample, but the device still holds 2**n amplitudes.

    So angle's host pool is far smaller than amplitude's while their device
    footprints are identical -- a caller sizing only by input width would be wrong.
    """
    amplitude = estimate_memory(
        num_qubits=14, batch_size=8, encoding_method="amplitude"
    )
    angle = estimate_memory(num_qubits=14, batch_size=8, encoding_method="angle")

    assert angle.gpu_state_bytes == amplitude.gpu_state_bytes
    assert angle.cpu_prefetch_bytes < amplitude.cpu_prefetch_bytes


def test_the_estimate_grows_past_any_real_device(estimate_memory):
    """The rejection case: 30 qubits at batch 64 is ~1 TiB of state, on purpose.

    This is the configuration the construction guard turns into an immediate error;
    here it is just a number, which is what makes it checkable without a GPU.
    """
    est = estimate_memory(
        num_qubits=30, batch_size=64, encoding_method="amplitude", dtype="f32"
    )

    assert est.gpu_state_bytes == 2 * 64 * (1 << 30) * 8


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"encoding_method": "quantum-teapot"}, id="unknown-encoding"),
        pytest.param({"dtype": "float128"}, id="unknown-dtype"),
        pytest.param({"num_qubits": 64}, id="state-vector-not-representable"),
        pytest.param({"batch_size": 2**62, "num_qubits": 30}, id="overflow"),
    ],
)
def test_bad_arguments_raise_value_error(estimate_memory, kwargs):
    """All four failure modes are argument errors, so all four must be ValueError.

    A RuntimeError here would read as "the device failed", which is never the cause:
    nothing has been allocated at this point.
    """
    args = {"num_qubits": 16, "batch_size": 64}
    args.update(kwargs)

    with pytest.raises(ValueError):
        estimate_memory(**args)


def test_names_are_case_insensitive_like_the_rest_of_the_api(estimate_memory):
    """Loader and engine accept 'Float32'/'AMPLITUDE'; this must not be the exception."""
    upper = estimate_memory(
        num_qubits=10, batch_size=4, encoding_method="AMPLITUDE", dtype="Float32"
    )
    lower = estimate_memory(
        num_qubits=10, batch_size=4, encoding_method="amplitude", dtype="f32"
    )

    assert upper == lower


def test_estimate_is_exported_from_the_package_root():
    """Users are told to ``from qumat_qdp import estimate_memory``; keep that true."""
    import qumat_qdp

    assert "estimate_memory" in qumat_qdp.__all__
    assert "MemoryEstimate" in qumat_qdp.__all__
    assert isinstance(
        qumat_qdp.estimate_memory(num_qubits=4, batch_size=2),
        qumat_qdp.MemoryEstimate,
    )
