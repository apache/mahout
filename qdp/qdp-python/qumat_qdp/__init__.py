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

Public API: QdpEngine (unified router), QdpTensor/QuantumTensor (DLPack facade),
QdpBenchmark, ThroughputResult, LatencyResult (benchmark API),
QuantumDataLoader (data loader iterator).

Usage:
    from qumat_qdp import QdpEngine, QuantumTensor
    from qumat_qdp import QdpBenchmark, ThroughputResult, LatencyResult
    from qumat_qdp import QuantumDataLoader
"""

from __future__ import annotations

# Backend detection: gracefully degrade when _qdp (Rust extension) is unavailable.
from qumat_qdp._backend import Backend, force_backend, get_backend, get_qdp

_qdp_mod = get_qdp()
if _qdp_mod is not None:
    RustQdpEngine = getattr(_qdp_mod, "QdpEngine")
    NativeQuantumTensor = getattr(_qdp_mod, "QuantumTensor")
    run_throughput_pipeline_py = getattr(_qdp_mod, "run_throughput_pipeline_py", None)
else:
    RustQdpEngine = None
    NativeQuantumTensor = None
    run_throughput_pipeline_py = None

BACKEND = get_backend()


def is_cuda_available() -> bool:
    """Return whether a usable CUDA device is available to the native engine.

    Unlike the importability of the ``_qdp`` extension -- which only means it
    was built, possibly against CUDA stubs on a host without the toolkit -- this
    reflects whether GPU work can actually run: ``False`` for a stub build or a
    host with no CUDA device, ``True`` only when the native runtime reports at
    least one device.
    """
    if _qdp_mod is None:
        return False
    probe = getattr(_qdp_mod, "cuda_available", None)
    if probe is None and hasattr(_qdp_mod, "_qdp"):
        probe = getattr(_qdp_mod._qdp, "cuda_available", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:
        return False


from qumat_qdp.api import (
    LatencyResult,
    QdpBenchmark,
    ThroughputResult,
)
from qumat_qdp.backend import QdpEngine
from qumat_qdp.loader import QuantumDataLoader
from qumat_qdp.tensor import QdpTensor, QuantumTensor
from qumat_qdp.triton_amd import TritonAmdEngine, is_triton_amd_available

__all__ = [
    "BACKEND",
    "Backend",
    "LatencyResult",
    "NativeQuantumTensor",
    "QdpBenchmark",
    "QdpEngine",
    "QdpTensor",
    "QuantumDataLoader",
    "QuantumTensor",
    "RustQdpEngine",
    "ThroughputResult",
    "TritonAmdEngine",
    "force_backend",
    "is_cuda_available",
    "is_triton_amd_available",
    "run_throughput_pipeline_py",
]
