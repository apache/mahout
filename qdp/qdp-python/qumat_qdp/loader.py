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
Quantum Data Loader: Python builder for Rust-backed batch iterator.

Usage:
    from qumat_qdp import QuantumDataLoader

    loader = (QuantumDataLoader(device_id=0).qubits(16).encoding("amplitude")
              .batches(100, size=64).source_synthetic())
    for qt in loader:
        batch = torch.from_dlpack(qt)
        ...
"""

from __future__ import annotations

from typing import Iterator, Optional

# Lazy import _qdp until __iter__ is used
_qdp: Optional[object] = None


def _get_qdp():
    global _qdp
    if _qdp is not None:
        return _qdp
    import _qdp as m

    _qdp = m
    return m


class QuantumDataLoader:
    """
    Builder for a synthetic-data quantum encoding iterator.

    Yields one QuantumTensor (batch) per iteration. All encoding runs in Rust;
    __iter__ returns the Rust-backed iterator from create_synthetic_loader.
    """

    def __init__(
        self,
        device_id: int = 0,
        num_qubits: int = 16,
        batch_size: int = 64,
        total_batches: int = 100,
        encoding_method: str = "amplitude",
        seed: Optional[int] = None,
    ) -> None:
        self._device_id = device_id
        self._num_qubits = num_qubits
        self._batch_size = batch_size
        self._total_batches = total_batches
        self._encoding_method = encoding_method
        self._seed = seed

    def qubits(self, n: int) -> QuantumDataLoader:
        """Set number of qubits. Returns self for chaining."""
        self._num_qubits = n
        return self

    def encoding(self, method: str) -> QuantumDataLoader:
        """Set encoding method (e.g. 'amplitude', 'angle', 'basis'). Returns self."""
        self._encoding_method = method
        return self

    def batches(self, total: int, size: int = 64) -> QuantumDataLoader:
        """Set total number of batches and batch size. Returns self."""
        self._total_batches = total
        self._batch_size = size
        return self

    def source_synthetic(
        self,
        total_batches: Optional[int] = None,
    ) -> QuantumDataLoader:
        """Use synthetic data source (default). Optionally override total_batches. Returns self."""
        if total_batches is not None:
            self._total_batches = total_batches
        return self

    def seed(self, s: Optional[int] = None) -> QuantumDataLoader:
        """Set RNG seed for reproducible synthetic data. Returns self."""
        self._seed = s
        return self

    def __iter__(self) -> Iterator[object]:
        """Return Rust-backed iterator that yields one QuantumTensor per batch."""
        qdp = _get_qdp()
        QdpEngine = getattr(qdp, "QdpEngine", None)
        if QdpEngine is None:
            raise RuntimeError(
                "_qdp.QdpEngine not found. Build the extension with maturin develop."
            )
        engine = QdpEngine(device_id=self._device_id)
        create_synthetic_loader = getattr(engine, "create_synthetic_loader", None)
        if create_synthetic_loader is None:
            raise RuntimeError(
                "create_synthetic_loader not available (e.g. only on Linux with CUDA)."
            )
        loader = create_synthetic_loader(
            total_batches=self._total_batches,
            batch_size=self._batch_size,
            num_qubits=self._num_qubits,
            encoding_method=self._encoding_method,
            seed=self._seed,
        )
        return iter(loader)
