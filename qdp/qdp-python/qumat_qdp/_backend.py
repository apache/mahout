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

"""
Backend detection and selection for QDP.

Priority order:
1. _qdp (Rust+CUDA) -- native extension, highest performance
2. torch (PyTorch) -- fallback, optional dependency
3. None -- neither available
"""

from __future__ import annotations

import enum
from functools import lru_cache
from types import ModuleType


class Backend(enum.Enum):
    """Available QDP encoding backends."""

    RUST_CUDA = "rust_cuda"
    PYTORCH = "pytorch"
    NONE = "none"


# Module-level override; set via force_backend().
_forced_backend: Backend | None = None


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


def get_backend() -> Backend:
    """Return the highest-priority available backend.

    Respects :func:`force_backend` overrides.
    """
    if _forced_backend is not None:
        return _forced_backend
    if get_qdp() is not None:
        return Backend.RUST_CUDA
    if get_torch() is not None:
        return Backend.PYTORCH
    return Backend.NONE


def force_backend(backend: Backend | None) -> None:
    """Override automatic backend detection.

    Pass ``None`` to restore auto-detection.  Primarily useful for
    testing and benchmarking.
    """
    global _forced_backend
    _forced_backend = backend


def require_backend() -> Backend:
    """Return the current backend or raise if none is available."""
    b = get_backend()
    if b is Backend.NONE:
        raise RuntimeError(
            "No QDP encoding backend available. "
            "Install PyTorch (pip install torch) or build the Rust extension (maturin develop)."
        )
    return b
