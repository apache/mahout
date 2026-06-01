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
#
import numpy as np
import pytest

from qumat.neutrosophic import NeutroBit


def test_neutrobit_normalization_and_measurement_sum_to_one():
    bit = NeutroBit(1 + 0j, 1 + 0j, 1 + 0j).normalize()
    probabilities = bit.measurement_probabilities()
    assert pytest.approx(sum(probabilities.values())) == 1.0


def test_zero_vector_normalization_is_rejected():
    with pytest.raises(ValueError):
        NeutroBit(0j, 0j, 0j).normalize()


def test_projection_to_qubit_subspace_matches_expected_values():
    bit = NeutroBit(1 / np.sqrt(2), 1 / np.sqrt(2), 0)
    np.testing.assert_allclose(
        bit.project_to_qubit_subspace(),
        np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex),
    )


def test_qubit_projection_rejects_empty_projection_when_normalized():
    with pytest.raises(ValueError):
        NeutroBit(0, 0, 1).project_to_qubit_subspace()


def test_neutrobit_from_vector_accepts_normalized_construction():
    bit = NeutroBit.from_vector([2, 0, 0], normalize=True)
    np.testing.assert_allclose(bit.vector(), np.array([1, 0, 0], dtype=complex))
