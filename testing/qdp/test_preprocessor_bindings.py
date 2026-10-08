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


"""Tests for the raw ``_qdp.Preprocessor`` PyO3 binding (CPU-only).

These tests pin down the native layer: argument conversion, error
mapping, and memory layout handling. No GPU is needed (works on a CUDA-stub build).
"""

import numpy as np
import pytest


@pytest.fixture
def native():
    import _qdp

    return _qdp.Preprocessor


def test_static_methods_exist(native):
    for name in (
        "validate_input",
        "calculate_l2_norm",
        "validate_batch",
        "calculate_batch_l2_norms",
    ):
        assert callable(getattr(native, name))


def test_l2_norm_list_and_array(native):
    assert native.calculate_l2_norm([3.0, 4.0]) == pytest.approx(5.0)
    assert native.calculate_l2_norm(np.array([3.0, 4.0])) == pytest.approx(5.0)


def test_int_and_float32_inputs_are_converted(native):
    assert native.calculate_l2_norm([3, 4]) == pytest.approx(5.0)
    x = np.array([3.0, 4.0], dtype=np.float32)
    assert native.calculate_l2_norm(x) == pytest.approx(5.0)


def test_core_errors_map_to_runtime_error(native):
    with pytest.raises(RuntimeError, match="Invalid input: Input data has zero norm"):
        native.calculate_l2_norm([0.0, 0.0])
    with pytest.raises(RuntimeError, match="Invalid input: .*exceeds state vector"):
        native.validate_input([1.0, 2.0, 3.0], 1)


def test_conversion_errors(native):
    with pytest.raises(TypeError):
        native.calculate_l2_norm(np.ones((2, 2)))  # wrong rank
    with pytest.raises(ValueError, match="could not convert"):
        native.calculate_l2_norm(["a", "b"])


def test_fortran_order_batch_is_read_row_major(native):
    batch = np.asfortranarray([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    np.testing.assert_allclose(native.calculate_batch_l2_norms(batch, 2, 3), [5.0, 2.0])


def test_zero_sample_size_does_not_abort(native):
    # Without the binding-side guard this reaches rayon's par_chunks(0),
    # which panics; the release profile uses panic = "abort".
    with pytest.raises(RuntimeError, match="Invalid input"):
        native.calculate_batch_l2_norms(np.ones(4), 2, 0)
