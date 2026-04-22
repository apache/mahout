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

"""CUDA backend adapter backed by the native ``_qdp`` extension."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _load_qdp_module() -> Any | None:
    try:
        return import_module("_qdp")
    except ImportError:
        return None


_qdp_mod = _load_qdp_module()


class _MissingCudaBackendEngine:
    def __init__(self, device_id: int = 0, precision: str = "float32") -> None:
        raise RuntimeError(
            "_qdp.QdpEngine is unavailable. Build the extension with: maturin develop"
        )


CudaBackendEngine = getattr(_qdp_mod, "QdpEngine", _MissingCudaBackendEngine)
