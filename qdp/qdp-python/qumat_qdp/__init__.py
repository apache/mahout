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
QDP (Quantum Data Processing) Python API.

Public API: QdpEngine, AmdQdpEngine, QuantumTensor (Rust extension _qdp),
QdpBenchmark, ThroughputResult, LatencyResult (benchmark API),
QuantumDataLoader (data loader iterator), EngineRouter with backends:
CUDA and Triton AMD.

Usage:
    from qumat_qdp import QdpEngine, AmdQdpEngine, QuantumTensor
    from qumat_qdp import TritonAmdKernel, create_encoder_engine
    from qumat_qdp import QdpBenchmark, ThroughputResult, LatencyResult
    from qumat_qdp import QuantumDataLoader
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Literal, overload

# Backend detection: gracefully degrade when _qdp (Rust extension) is unavailable.
from qumat_qdp._backend import Backend, force_backend, get_backend, get_qdp

_qdp_mod: ModuleType | None = get_qdp()
if _qdp_mod is not None:
    QdpEngine = getattr(_qdp_mod, "QdpEngine", None)
    AmdQdpEngine = getattr(_qdp_mod, "AmdQdpEngine", None)
    QuantumTensor = getattr(_qdp_mod, "QuantumTensor", None)
    run_throughput_pipeline_py = getattr(_qdp_mod, "run_throughput_pipeline_py", None)
else:
    QdpEngine = None
    AmdQdpEngine = None
    QuantumTensor = None
    run_throughput_pipeline_py = None

BACKEND = get_backend()

from qumat_qdp.api import (
    LatencyResult,
    QdpBenchmark,
    ThroughputResult,
)
from qumat_qdp.backend import (
    EngineRouter,
    create_encoder_engine,
)
from qumat_qdp.loader import QuantumDataLoader

if TYPE_CHECKING:
    from qumat_qdp.triton_amd import TritonAmdKernel


def is_triton_amd_available() -> bool:
    try:
        from qumat_qdp.triton_amd import is_triton_amd_available as _fn

        return _fn()
    except Exception:
        return False


@overload
def __getattr__(name: Literal["TritonAmdKernel"]) -> type[TritonAmdKernel]: ...


@overload
def __getattr__(name: str) -> object: ...


def __getattr__(name: str) -> object:
    if name == "TritonAmdKernel":
        from qumat_qdp.triton_amd import TritonAmdKernel

        return TritonAmdKernel
    raise AttributeError(name)


__all__ = [
    "BACKEND",
    "AmdQdpEngine",
    "Backend",
    "EngineRouter",
    "LatencyResult",
    "QdpBenchmark",
    "QdpEngine",
    "QuantumDataLoader",
    "QuantumTensor",
    "ThroughputResult",
    "TritonAmdKernel",
    "create_encoder_engine",
    "force_backend",
    "is_triton_amd_available",
    "run_throughput_pipeline_py",
]
