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

import pytest
import numpy as np
from importlib import import_module

from ..utils import TESTING_BACKENDS
from ..utils.qumat_helpers import get_qumat_example_final_state_vector


@pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
class TestFinalQuantumStates:
    """Test class for final quantum state comparisons between QuMat and native implementations."""

    @pytest.mark.parametrize("initial_ket_str", ["000", "001", "010", "011"])
    def test_backend_final_state_vector(self, backend_name, initial_ket_str):
        """Test that QuMat produces same final state as native backend implementation."""
        # Import backend-specific helpers
        backend_module = import_module(
            f".{backend_name}_helpers", package="testing.utils"
        )

        # Get native implementation result
        native_example_vector = backend_module.get_native_example_final_state_vector(
            initial_ket_str
        )

        # Get QuMat implementation result
        qumat_backend_config = backend_module.get_qumat_backend_config(
            "get_final_state_vector"
        )
        qumat_example_vector = get_qumat_example_final_state_vector(
            qumat_backend_config, initial_ket_str
        )

        # Compare final state vectors from QuMat vs. native implementation
        np.testing.assert_array_equal(
            qumat_example_vector,
            native_example_vector,
            err_msg=f"State vectors don't match for initial state {initial_ket_str} using {backend_name}",
        )


class TestFinalQuantumStatesConsistency:
    """Test class for consistency checks across all backends."""

    def test_all_backends_consistency(self):
        """Test that all available backends produce consistent results."""
        initial_ket_str = "001"
        results = {}

        for backend_name in TESTING_BACKENDS:
            backend_module = import_module(
                f".{backend_name}_helpers", package="testing.utils"
            )
            qumat_backend_config = backend_module.get_qumat_backend_config(
                "get_final_state_vector"
            )
            results[backend_name] = get_qumat_example_final_state_vector(
                qumat_backend_config, initial_ket_str
            )
