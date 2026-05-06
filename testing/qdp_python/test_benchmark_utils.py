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

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_QDP_PYTHON = Path(__file__).resolve().parents[2] / "qdp" / "qdp-python"
_UTILS_PATH = _QDP_PYTHON / "benchmark" / "utils.py"

_UTILS_SPEC = importlib.util.spec_from_file_location("qdp_benchmark_utils", _UTILS_PATH)
assert _UTILS_SPEC is not None
assert _UTILS_SPEC.loader is not None
_benchmark_utils = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(_benchmark_utils)

build_sample = _benchmark_utils.build_sample
generate_batch_data = _benchmark_utils.generate_batch_data
normalize_batch = _benchmark_utils.normalize_batch
normalize_batch_torch = _benchmark_utils.normalize_batch_torch


def test_build_sample_iqp_z_has_qubit_sized_parameter_vector():
    sample = build_sample(seed=5, vector_len=4, encoding_method="iqp-z")

    assert sample.shape == (4,)
    assert np.all(sample >= 0.0)
    assert np.all(sample < 2.0 * np.pi)


def test_build_sample_iqp_has_full_parameter_vector():
    sample = build_sample(seed=7, vector_len=10, encoding_method="iqp")

    assert sample.shape == (10,)
    assert np.all(sample >= 0.0)
    assert np.all(sample < 2.0 * np.pi)


@pytest.mark.parametrize("encoding_method", ["angle", "iqp", "iqp-z"])
def test_build_sample_phase_encodings_handle_zero_length(encoding_method):
    sample = build_sample(seed=3, vector_len=0, encoding_method=encoding_method)

    assert sample.shape == (0,)


def test_generate_batch_data_iqp_family_uses_phase_ranges():
    iqp = generate_batch_data(3, 6, encoding_method="iqp", seed=11)
    iqp_z = generate_batch_data(3, 4, encoding_method="iqp-z", seed=11)

    assert iqp.shape == (3, 6)
    assert iqp_z.shape == (3, 4)
    assert np.all(iqp >= 0.0)
    assert np.all(iqp < 2.0 * np.pi)
    assert np.all(iqp_z >= 0.0)
    assert np.all(iqp_z < 2.0 * np.pi)


def test_normalize_batch_leaves_iqp_family_unchanged():
    batch = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float64)

    assert np.array_equal(normalize_batch(batch, "iqp"), batch)
    assert np.array_equal(normalize_batch(batch, "iqp-z"), batch)


def test_normalize_batch_torch_leaves_iqp_family_unchanged():
    batch = torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=torch.float64)

    assert torch.equal(normalize_batch_torch(batch, "iqp"), batch)
    assert torch.equal(normalize_batch_torch(batch, "iqp-z"), batch)
