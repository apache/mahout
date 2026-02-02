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

from functools import lru_cache
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    import _qdp  # noqa: F401 -- for type checkers only

# Rust interface expects seed as Option<u64>: non-negative and <= 2^64 - 1.
# Ref: qdp-core PipelineConfig seed: Option<u64>
_U64_MAX = 2**64 - 1

# Lazy import _qdp at runtime until __iter__ is used; TYPE_CHECKING import above
# is for type checkers only so they can resolve "_qdp.*" annotations if needed.


@lru_cache(maxsize=1)
def _get_qdp():
    import _qdp as m

    return m


def _validate_loader_args(
    *,
    device_id: int,
    num_qubits: int,
    batch_size: int,
    total_batches: int,
    encoding_method: str,
    seed: Optional[int],
) -> None:
    """Validate arguments before passing to Rust (PipelineConfig / create_synthetic_loader)."""
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id!r}")
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError(f"num_qubits must be a positive integer, got {num_qubits!r}")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not isinstance(total_batches, int) or total_batches < 1:
        raise ValueError(
            f"total_batches must be a positive integer, got {total_batches!r}"
        )
    if not encoding_method or not isinstance(encoding_method, str):
        raise ValueError(
            f"encoding_method must be a non-empty string, got {encoding_method!r}"
        )
    if seed is not None:
        if not isinstance(seed, int):
            raise ValueError(
                f"seed must be None or an integer, got {type(seed).__name__!r}"
            )
        if seed < 0 or seed > _U64_MAX:
            raise ValueError(
                f"seed must be in range [0, {_U64_MAX}] (Rust u64), got {seed!r}"
            )


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
        _validate_loader_args(
            device_id=device_id,
            num_qubits=num_qubits,
            batch_size=batch_size,
            total_batches=total_batches,
            encoding_method=encoding_method,
            seed=seed,
        )
        self._device_id = device_id
        self._num_qubits = num_qubits
        self._batch_size = batch_size
        self._total_batches = total_batches
        self._encoding_method = encoding_method
        self._seed = seed

    def qubits(self, n: int) -> QuantumDataLoader:
        """Set number of qubits. Returns self for chaining."""
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"num_qubits must be a positive integer, got {n!r}")
        self._num_qubits = n
        return self

    def encoding(self, method: str) -> QuantumDataLoader:
        """Set encoding method (e.g. 'amplitude', 'angle', 'basis'). Returns self."""
        if not method or not isinstance(method, str):
            raise ValueError(
                f"encoding_method must be a non-empty string, got {method!r}"
            )
        self._encoding_method = method
        return self

    def batches(self, total: int, size: int = 64) -> QuantumDataLoader:
        """Set total number of batches and batch size. Returns self."""
        if not isinstance(total, int) or total < 1:
            raise ValueError(f"total_batches must be a positive integer, got {total!r}")
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {size!r}")
        self._total_batches = total
        self._batch_size = size
        return self

    def source_synthetic(
        self,
        total_batches: Optional[int] = None,
    ) -> QuantumDataLoader:
        """Use synthetic data source (default). Optionally override total_batches. Returns self."""
        if total_batches is not None:
            if not isinstance(total_batches, int) or total_batches < 1:
                raise ValueError(
                    f"total_batches must be a positive integer, got {total_batches!r}"
                )
            self._total_batches = total_batches
        return self

    def seed(self, s: Optional[int] = None) -> QuantumDataLoader:
        """Set RNG seed for reproducible synthetic data (must fit Rust u64: 0 <= seed <= 2^64-1). Returns self."""
        if s is not None:
            if not isinstance(s, int):
                raise ValueError(
                    f"seed must be None or an integer, got {type(s).__name__!r}"
                )
            if s < 0 or s > _U64_MAX:
                raise ValueError(
                    f"seed must be in range [0, {_U64_MAX}] (Rust u64), got {s!r}"
                )
        self._seed = s
        return self

    def __iter__(self) -> Iterator[object]:
        """Return Rust-backed iterator that yields one QuantumTensor per batch."""
        _validate_loader_args(
            device_id=self._device_id,
            num_qubits=self._num_qubits,
            batch_size=self._batch_size,
            total_batches=self._total_batches,
            encoding_method=self._encoding_method,
            seed=self._seed,
        )
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
