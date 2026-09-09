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
from qumat_qdp import QuantumDataLoader


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Amplitude", "amplitude"),
        ("ANGLE", "angle"),
        ("IQP-Z", "iqp-z"),
        ("Basis", "basis"),
    ],
)
def test_encoding_setter_normalizes_case(given: str, expected: str) -> None:
    loader = QuantumDataLoader().encoding(given)
    assert loader._encoding_method == expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Amplitude", "amplitude"),
        ("ANGLE", "angle"),
        ("IQP-Z", "iqp-z"),
        ("Basis", "basis"),
    ],
)
def test_constructor_normalizes_encoding_case(given: str, expected: str) -> None:
    loader = QuantumDataLoader(encoding_method=given)
    assert loader._encoding_method == expected


def test_mixed_case_encoding_iterates_without_error() -> None:
    pytest.importorskip("torch")
    loader = (
        QuantumDataLoader()
        .backend("pytorch")
        .qubits(2)
        .encoding("Amplitude")
        .batches(1, size=1)
    )
    batches = list(loader)
    assert len(batches) == 1


@pytest.mark.parametrize("bad", ["not-an-encoding", "amplitud", "ANGL"])
def test_unknown_encoding_still_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        QuantumDataLoader().encoding(bad)
