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
Quantum Data Plane (QDP) - GPU-accelerated quantum state encoding.

This module provides a unified interface to the QDP engine, enabling
GPU-accelerated encoding of classical data into quantum states with
zero-copy PyTorch integration via DLPack.

Example:
    >>> import qumat.qdp as qdp
    >>> engine = qdp.QdpEngine(device_id=0)
    >>> qtensor = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2, encoding_method="amplitude")
    >>> import torch
    >>> torch_tensor = torch.from_dlpack(qtensor)
"""

_INSTALL_MSG = (
    "QDP requires the qumat-qdp native extension. "
    "Build and install it with: cd qdp/qdp-python && maturin develop --release"
)


def _make_stub(name: str) -> type:
    """Create a stub class that raises ImportError on instantiation."""

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError(_INSTALL_MSG)

    return type(name, (), {"__init__": __init__, "__doc__": f"Stub class - {name}"})


try:
    from _qdp import QdpEngine as QdpEngine
    from _qdp import QuantumTensor as QuantumTensor
except ImportError as e:
    import warnings

    warnings.warn(
        f"QDP module not available: {e}. "
        "QDP requires the qumat-qdp native extension which needs to be built with maturin. "
        "See qdp/qdp-python/README.md for installation instructions.",
        ImportWarning,
    )

    QdpEngine = _make_stub("QdpEngine")  # type: ignore[misc]
    QuantumTensor = _make_stub("QuantumTensor")  # type: ignore[misc]

__all__ = ["QdpEngine", "QuantumTensor"]
