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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_complex_vector(state: Any) -> np.ndarray:
    return np.asarray(state, dtype=complex).reshape(3)


@dataclass(frozen=True)
class NeutroBit:
    """Three-state experimental carrier over |0>, |1>, and |I>."""

    alpha: complex
    beta: complex
    gamma: complex

    def vector(self) -> np.ndarray:
        return np.array([self.alpha, self.beta, self.gamma], dtype=complex)

    def norm_squared(self) -> float:
        return float(np.sum(np.abs(self.vector()) ** 2).real)

    def normalize(self) -> "NeutroBit":
        norm = np.sqrt(self.norm_squared())
        if np.isclose(norm, 0.0):
            raise ValueError("Cannot normalize the zero neutrobit.")
        normalized = self.vector() / norm
        return NeutroBit(*normalized.tolist())

    def measurement_probabilities(self) -> dict[str, float]:
        normalized = self.normalize().vector()
        probabilities = np.abs(normalized) ** 2
        return {
            "|0>": float(probabilities[0].real),
            "|1>": float(probabilities[1].real),
            "|I>": float(probabilities[2].real),
        }

    def project_to_qubit_subspace(self, normalize: bool = True) -> np.ndarray:
        vector = self.vector()[:2].astype(complex)
        if not normalize:
            return vector
        norm = np.sqrt(float(np.sum(np.abs(vector) ** 2).real))
        if np.isclose(norm, 0.0):
            raise ValueError("Cannot normalize an empty qubit projection.")
        return vector / norm

    @classmethod
    def from_vector(cls, state: Any, normalize: bool = False) -> "NeutroBit":
        vector = _as_complex_vector(state)
        bit = cls(*vector.tolist())
        return bit.normalize() if normalize else bit
