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

from qumat_qdp import create_encoder_engine
from qumat_qdp.backend import QuantumTensor


def test_backend_routing_rejects_unknown_backend():
    with pytest.raises(ValueError):
        create_encoder_engine(backend="unknown-backend")


def test_auto_router_without_available_backends_fails_cleanly():
    # On environments without _qdp / Triton accelerators, auto should fail
    # with an actionable runtime error (instead of importing CUDA eagerly).
    try:
        create_encoder_engine(backend="auto")
    except RuntimeError as exc:
        assert "No available backend" in str(exc)


def test_unified_quantum_tensor_requires_dlpack():
    with pytest.raises(RuntimeError):
        QuantumTensor(value=object(), backend="fake").__dlpack__()
