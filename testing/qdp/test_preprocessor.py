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

"""Tests for qumat_qdp.Preprocessor (CPU-only; no GPU required).

These need the _qdp extension (auto-skipped by testing/conftest.py when it is
missing) but never touch CUDA, so they run on a stub build too.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import qumat_qdp
from qumat_qdp import Preprocessor

REPO_ROOT = Path(__file__).resolve().parents[2]
INVALID = r"Invalid input"


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_preprocessor_exported_in_all():
    assert "Preprocessor" in qumat_qdp.__all__
    assert qumat_qdp.Preprocessor is Preprocessor


@pytest.mark.parametrize(
    "name",
    [
        "validate_input",
        "calculate_l2_norm",
        "validate_batch",
        "calculate_batch_l2_norms",
    ],
)
def test_public_methods_have_docstrings(name):
    doc = getattr(Preprocessor, name).__doc__
    assert doc, f"Preprocessor.{name} has no docstring"
    assert ":raises RuntimeError:" in doc


def test_native_class_registered():
    import _qdp

    assert hasattr(_qdp, "Preprocessor")


def test_missing_extension_raises_clear_error(monkeypatch):
    import qumat_qdp.preprocessing as mod

    monkeypatch.setattr(mod, "get_qdp", lambda: None)
    with pytest.raises(RuntimeError, match="requires the _qdp Rust extension"):
        Preprocessor.calculate_l2_norm([1.0])


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "num_qubits"),
    [
        ([1.0], 1),
        ([1.0, 2.0], 1),
        ([1.0, 2.0, 3.0], 2),  # shorter than 2**n is fine (zero-padded)
        (np.ones(8), 3),
        (np.ones(4, dtype=np.float32), 2),  # converted to float64
        ((1, 2, 3, 4), 2),  # tuple of ints
    ],
)
def test_validate_input_accepts(data, num_qubits):
    assert Preprocessor.validate_input(data, num_qubits) is None


@pytest.mark.parametrize(
    ("data", "num_qubits", "message"),
    [
        ([1.0], 0, "at least 1"),
        ([1.0], 31, "exceeds practical limit"),
        ([], 2, "cannot be empty"),
        (np.ones(5), 2, "exceeds state vector size 4"),
    ],
)
def test_validate_input_rejects(data, num_qubits, message):
    with pytest.raises(RuntimeError, match=f"{INVALID}.*{message}"):
        Preprocessor.validate_input(data, num_qubits)


def test_validate_input_rejects_2d():
    with pytest.raises(TypeError):
        Preprocessor.validate_input(np.ones((2, 2)), 2)


def test_validate_input_rejects_non_numeric():
    with pytest.raises(ValueError, match="could not convert"):
        Preprocessor.validate_input(["a"], 1)


# ---------------------------------------------------------------------------
# calculate_l2_norm
# ---------------------------------------------------------------------------


def test_calculate_l2_norm_value():
    assert Preprocessor.calculate_l2_norm([3.0, 4.0]) == pytest.approx(5.0)


def test_calculate_l2_norm_returns_float():
    assert isinstance(Preprocessor.calculate_l2_norm(np.array([1.0])), float)


def test_calculate_l2_norm_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024)
    assert Preprocessor.calculate_l2_norm(x) == pytest.approx(np.linalg.norm(x))


def test_calculate_l2_norm_non_contiguous_view():
    x = np.arange(1.0, 9.0)
    view = x[::2]  # [1, 3, 5, 7], not contiguous
    assert not view.flags["C_CONTIGUOUS"]
    expected = np.linalg.norm(view)
    assert Preprocessor.calculate_l2_norm(view) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([0.0, 0.0, 0.0], "zero norm"),
        ([], "zero norm"),
        ([1.0, float("nan")], "NaN"),
        ([1.0, float("inf")], "Infinity"),
    ],
)
def test_calculate_l2_norm_rejects(data, message):
    with pytest.raises(RuntimeError, match=f"{INVALID}.*{message}"):
        Preprocessor.calculate_l2_norm(data)


# ---------------------------------------------------------------------------
# validate_batch
# ---------------------------------------------------------------------------


def test_validate_batch_accepts_2d_and_flat():
    batch = np.ones((3, 4))
    assert Preprocessor.validate_batch(batch, 3, 4, 2) is None
    assert Preprocessor.validate_batch(batch.ravel(), 3, 4, 2) is None
    assert Preprocessor.validate_batch(batch.ravel().tolist(), 3, 4, 2) is None


@pytest.mark.parametrize(
    ("batch", "num_samples", "sample_size", "num_qubits", "message"),
    [
        (np.ones(4), 0, 4, 2, "num_samples must be greater than 0"),
        (np.ones(7), 2, 4, 2, "doesn't match num_samples 2 \\* sample_size 4"),
        (np.ones((2, 4)), 4, 2, 2, "shape \\(2, 4\\)"),
        (np.ones(8), 1, 8, 2, "Sample size 8 exceeds state vector size 4"),
        (np.ones(4), 1, 4, 0, "at least 1"),
        (np.ones((1, 1, 4)), 1, 4, 2, "must be 1D .* or 2D"),
    ],
)
def test_validate_batch_rejects(batch, num_samples, sample_size, num_qubits, message):
    with pytest.raises(RuntimeError, match=f"{INVALID}.*{message}"):
        Preprocessor.validate_batch(batch, num_samples, sample_size, num_qubits)


# ---------------------------------------------------------------------------
# calculate_batch_l2_norms
# ---------------------------------------------------------------------------


def test_calculate_batch_l2_norms_values():
    batch = np.array([[3.0, 4.0], [6.0, 8.0], [1.0, 0.0]])
    norms = Preprocessor.calculate_batch_l2_norms(batch, 3, 2)
    assert isinstance(norms, np.ndarray)
    assert norms.dtype == np.float64
    np.testing.assert_allclose(norms, [5.0, 10.0, 1.0])


def test_calculate_batch_l2_norms_flat_input():
    norms = Preprocessor.calculate_batch_l2_norms([3.0, 4.0, 6.0, 8.0], 2, 2)
    np.testing.assert_allclose(norms, [5.0, 10.0])


def test_calculate_batch_l2_norms_fortran_order_keeps_rows():
    """A Fortran-ordered 2D array must still be read row by row."""
    batch = np.asfortranarray([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    assert not batch.flags["C_CONTIGUOUS"]
    norms = Preprocessor.calculate_batch_l2_norms(batch, 2, 3)
    np.testing.assert_allclose(norms, [5.0, 2.0])


def test_calculate_batch_l2_norms_matches_numpy():
    rng = np.random.default_rng(1)
    batch = rng.standard_normal((16, 32))
    norms = Preprocessor.calculate_batch_l2_norms(batch, 16, 32)
    np.testing.assert_allclose(norms, np.linalg.norm(batch, axis=1))


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        ([[1.0, 1.0], [0.0, 0.0]], "Sample 1 has zero norm"),
        ([[float("nan"), 1.0], [1.0, 1.0]], "Sample 0 produced NaN norm"),
        ([[1.0, 1.0], [float("inf"), 1.0]], "Sample 1 produced Infinity norm"),
    ],
)
def test_calculate_batch_l2_norms_rejects_bad_samples(batch, message):
    with pytest.raises(RuntimeError, match=f"{INVALID}.*{message}"):
        Preprocessor.calculate_batch_l2_norms(np.array(batch), 2, 2)


@pytest.mark.parametrize(
    ("batch", "num_samples", "sample_size"),
    [
        (np.ones(4), 2, 0),  # would panic in par_chunks(0) without the guard
        (np.ones(4), 0, 4),
        (np.ones(5), 2, 2),
    ],
)
def test_calculate_batch_l2_norms_rejects_bad_shape(batch, num_samples, sample_size):
    with pytest.raises(RuntimeError, match=INVALID):
        Preprocessor.calculate_batch_l2_norms(batch, num_samples, sample_size)


# ---------------------------------------------------------------------------
# The documented example must run as written.
# ---------------------------------------------------------------------------


def _doc_example() -> str:
    text = (REPO_ROOT / "docs" / "qdp" / "python-api.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    for block in blocks:
        if "Preprocessor.validate_input" in block:
            return block
    raise AssertionError("Preprocessor example not found in docs/qdp/python-api.md")


def test_docs_example_runs():
    namespace: dict = {}
    exec(compile(_doc_example(), "python-api.md", "exec"), namespace)
    amplitudes = namespace["amplitudes"]
    np.testing.assert_allclose(amplitudes, [0.6, 0.8])
    assert np.sum(amplitudes**2) == pytest.approx(1.0)
