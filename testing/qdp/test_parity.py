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

"""Cross-backend parity grid.

``qumat_qdp.torch_ref`` is the specification for every encoding. This module
runs the native Rust+CUDA engine over the full grid of

    encoding  x  engine precision  x  input shape  x  input location

and checks each cell against the reference. A new encoder is complete when it
has a reference function and every cell here passes. A refactor of the engine
or bindings is safe when this file is green before and after.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from .qdp_test_utils import requires_qdp

torch_ref = pytest.importorskip("qumat_qdp.torch_ref")

NUM_QUBITS = 4
BATCH = 5

ENCODINGS = ("amplitude", "angle", "basis", "iqp", "iqp-z", "phase")
PRECISIONS = ("float64", "float32")
SHAPES = ("single", "batch")
LOCATIONS = ("numpy", "torch_cpu", "cuda_f64", "cuda_f32")

# Encodings with a device-side float32 kernel today. Others reject f32 CUDA
# input, which is covered by ``test_f32_cuda_rejected_without_kernel``.
F32_DEVICE_ENCODINGS = ("amplitude", "angle", "basis")

# Encodings the current engine cannot read from a CUDA tensor at all. Cells
# for these are expected failures until the single generic encode path lands;
# once a listed encoding succeeds the cell fails with a message asking for the
# entry to be removed, so the list cannot go stale.
NO_DEVICE_PATH_ENCODINGS = ("phase",)

# A cell's tolerance follows the narrowest precision on its path: the engine's
# output precision, or float32 when the input arrives as a float32 CUDA tensor
# and is encoded by the float32 kernel before any widening.
TOL: dict[str, dict[str, float]] = {
    "float64": {"atol": 1e-10, "rtol": 1e-9},
    "float32": {"atol": 2e-5, "rtol": 2e-5},
}


def sample_dim(encoding: str, n: int) -> int:
    if encoding == "amplitude":
        return 1 << n
    if encoding == "basis":
        return 1
    if encoding == "iqp":
        return n + n * (n - 1) // 2
    return n  # angle, phase, iqp-z


def make_samples(encoding: str, n: int, batch: int) -> np.ndarray:
    """Deterministic float64 inputs of shape (batch, sample_dim)."""
    rng = np.random.default_rng(1234 + len(encoding))
    dim = sample_dim(encoding, n)
    if encoding == "amplitude":
        return rng.uniform(0.1, 1.0, size=(batch, dim))
    if encoding == "basis":
        return rng.integers(0, 1 << n, size=(batch, dim)).astype(np.float64)
    return rng.uniform(0.0, 2.0 * math.pi, size=(batch, dim))


def reference(encoding: str, data_2d: np.ndarray, n: int) -> torch.Tensor:
    x = torch.from_numpy(data_2d).to(torch.float64)
    return torch_ref.encode(x, n, encoding)


def to_input(encoding: str, data_2d: np.ndarray, shape: str, location: str):
    """Materialise the reference data as the requested input object."""
    arr = data_2d if shape == "batch" else data_2d[0]
    if location == "numpy":
        return np.ascontiguousarray(arr)
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if location == "torch_cpu":
        return t
    if location == "cuda_f64":
        # Basis indices on the device are int64 by API contract; every other
        # encoding takes float64.
        return (t.to(torch.int64) if encoding == "basis" else t).cuda()
    if location == "cuda_f32":
        return t.to(torch.float32).cuda()
    raise AssertionError(location)


def engine_output(qt) -> torch.Tensor:
    out = torch.from_dlpack(qt)
    assert out.is_cuda
    return out


def cell_id(p):
    return str(p)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("location", LOCATIONS, ids=cell_id)
@pytest.mark.parametrize("shape", SHAPES, ids=cell_id)
@pytest.mark.parametrize("precision", PRECISIONS, ids=cell_id)
@pytest.mark.parametrize("encoding", ENCODINGS, ids=cell_id)
def test_parity_grid(encoding, precision, shape, location):
    from _qdp import QdpEngine

    if location == "cuda_f32" and encoding not in F32_DEVICE_ENCODINGS:
        pytest.skip(f"{encoding} has no float32 device kernel")

    engine = QdpEngine(0, precision)
    data = make_samples(encoding, NUM_QUBITS, BATCH)
    # The engine always returns a (samples, 2**n) tensor, so a single sample
    # is compared against the first row of the batch reference.
    expected = reference(encoding, data, NUM_QUBITS)
    if shape == "single":
        expected = expected[:1]

    inp = to_input(encoding, data, shape, location)
    if location == "cuda_f64" and encoding in NO_DEVICE_PATH_ENCODINGS:
        try:
            engine.encode(inp, NUM_QUBITS, encoding)
        except RuntimeError:
            pytest.xfail(f"{encoding} has no device-pointer path yet")
        pytest.fail(
            f"{encoding} now encodes from a CUDA tensor; "
            "remove it from NO_DEVICE_PATH_ENCODINGS"
        )
    out = engine_output(engine.encode(inp, NUM_QUBITS, encoding))

    expected_dtype = torch.complex128 if precision == "float64" else torch.complex64
    assert out.dtype == expected_dtype
    assert tuple(out.shape) == tuple(expected.shape)

    # Float32 inputs on the device lose input precision before the kernel runs;
    # compare against the reference evaluated on the same rounded inputs.
    if location == "cuda_f32":
        rounded = data.astype(np.float32).astype(np.float64)
        expected = reference(encoding, rounded, NUM_QUBITS)
        if shape == "single":
            expected = expected[:1]

    tol = TOL["float32"] if location == "cuda_f32" else TOL[precision]
    torch.testing.assert_close(
        out.cpu().to(torch.complex128),
        expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
    )


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize(
    "encoding", [e for e in ENCODINGS if e not in F32_DEVICE_ENCODINGS]
)
def test_f32_cuda_rejected_without_kernel(encoding):
    """Encodings without an f32 device kernel must fail loudly, not silently cast."""
    from _qdp import QdpEngine

    engine = QdpEngine(0, "float32")
    data = make_samples(encoding, NUM_QUBITS, BATCH)
    inp = to_input(encoding, data, "batch", "cuda_f32")
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        engine.encode(inp, NUM_QUBITS, encoding)


@requires_qdp
@pytest.mark.gpu
@pytest.mark.parametrize("encoding", ENCODINGS, ids=cell_id)
def test_state_is_normalised(encoding):
    from _qdp import QdpEngine

    engine = QdpEngine(0, "float64")
    data = make_samples(encoding, NUM_QUBITS, BATCH)
    out = engine_output(engine.encode(data, NUM_QUBITS, encoding)).cpu()
    norms = (out.abs() ** 2).sum(dim=1)
    torch.testing.assert_close(
        norms, torch.ones(BATCH, dtype=norms.dtype), atol=1e-9, rtol=0
    )
