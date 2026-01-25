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

"""Shared pytest fixtures and markers for QDP tests."""

import pytest

QDP_SKIP_REASON = (
    "QDP extension not built. Run: uv run --active maturin develop "
    "--manifest-path qdp/qdp-python/Cargo.toml"
)


def _qdp_available():
    """Check if QDP extension is available."""
    try:
        import _qdp

        return _qdp is not None
    except ImportError:
        return False


requires_qdp = pytest.mark.skipif(not _qdp_available(), reason=QDP_SKIP_REASON)
