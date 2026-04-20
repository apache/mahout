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

"""Backend availability helpers for QDP."""

from __future__ import annotations

import enum
from functools import lru_cache
from types import ModuleType


class Backend(enum.Enum):
    """Available QDP backends exposed by the Python facade."""

    CUDA = "cuda"
    AMD = "amd"
    NONE = "none"


@lru_cache(maxsize=1)
def get_qdp() -> ModuleType | None:
    """Return the ``_qdp`` Rust extension module, or ``None`` if unavailable."""
    try:
        import _qdp as m

        return m
    except ImportError:
        return None


@lru_cache(maxsize=1)
def get_torch() -> ModuleType | None:
    """Return the ``torch`` module, or ``None`` if unavailable."""
    try:
        import torch as m

        return m
    except ImportError:
        return None


def get_default_backend() -> Backend:
    """Return the default backend for the current environment."""
    if get_qdp() is not None:
        return Backend.CUDA
    return Backend.NONE


def force_backend(backend: Backend | None) -> None:
    """Backward-compatible no-op.

    Backend selection is now explicit in ``qumat_qdp.QdpEngine(..., backend=...)``.
    This function remains to avoid breaking existing imports.
    """
    if backend is not None and backend not in (Backend.CUDA, Backend.AMD, Backend.NONE):
        raise ValueError(f"Unsupported backend override: {backend!r}")


def require_backend() -> Backend:
    """Return the default available backend or raise if none is available."""
    backend = get_default_backend()
    if backend is Backend.NONE:
        raise RuntimeError(
            "No QDP backend available. Build the Rust CUDA extension for `backend=\"cuda\"`, "
            "or install the ROCm/Triton runtime for `backend=\"amd\"`."
        )
    return backend
