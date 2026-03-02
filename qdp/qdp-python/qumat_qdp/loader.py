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
from typing import TYPE_CHECKING, Any, Iterator, Optional

import numpy as np

if TYPE_CHECKING:
    import _qdp  # noqa: F401 -- for type checkers only

# Optional torch for as_torch()/as_numpy(); import at use site to avoid hard dependency.
try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]

# Seed must fit Rust u64: 0 <= seed <= 2^64 - 1.
_U64_MAX = 2**64 - 1


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
        self._file_path: Optional[str] = None
        self._streaming_requested = (
            False  # set True by source_file(streaming=True); Phase 2b
        )
        self._synthetic_requested = False  # set True only by source_synthetic()
        self._file_requested = False
        self._array: Optional[np.ndarray] = None
        self._array_requested = False
        # Output format: None = yield raw QuantumTensor (DLPack); ("torch", device) or ("numpy",)
        self._output_format: Optional[tuple[str, ...]] = None

    def as_torch(self, device: str = "cuda") -> QuantumDataLoader:
        """Yield batches as PyTorch tensors. device='cuda' keeps data on GPU (no copy); 'cpu' moves to CPU. Returns self."""
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")
        if _torch is None:
            raise RuntimeError(
                "PyTorch is required for as_torch(). Install with: pip install torch"
            )
        self._output_format = ("torch", device)
        return self

    def as_numpy(self) -> QuantumDataLoader:
        """Yield batches as NumPy arrays (CPU). Conversion is done inside the loader. Returns self."""
        self._output_format = ("numpy",)
        return self

    def source_array(self, X: np.ndarray) -> QuantumDataLoader:
        """Use in-memory array; no temp file. Encodes via QdpEngine.encode() per batch. Returns self."""
        if X is None or not hasattr(X, "shape") or len(X.shape) != 2:
            raise ValueError(
                "source_array(X) requires a 2D array (n_samples, n_features)."
            )
        self._array = np.asarray(X, dtype=np.float64)
        if not self._array.flags.c_contiguous:
            self._array = np.ascontiguousarray(self._array)
        self._array_requested = True
        n = self._array.shape[0]
        self._total_batches = max(1, (n + self._batch_size - 1) // self._batch_size)
        return self

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
        self._synthetic_requested = True
        if total_batches is not None:
            if not isinstance(total_batches, int) or total_batches < 1:
                raise ValueError(
                    f"total_batches must be a positive integer, got {total_batches!r}"
                )
            self._total_batches = total_batches
        return self

    def source_file(self, path: str, streaming: bool = False) -> QuantumDataLoader:
        """Use file data source. Path must point to a supported format. Returns self.

        For streaming=True (Phase 2b), only .parquet is supported; data is read in chunks to reduce memory.
        For streaming=False, supports .parquet, .arrow, .feather, .ipc, .npy, .pt, .pth, .pb.
        """
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path!r}")
        if streaming and not (path.lower().endswith(".parquet")):
            raise ValueError(
                "streaming=True supports only .parquet files; use streaming=False for other formats."
            )
        self._file_path = path
        self._file_requested = True
        self._streaming_requested = streaming
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

    def _array_iterator(self) -> Iterator[Any]:
        """Yield one QuantumTensor per batch from in-memory array via QdpEngine.encode()."""
        qdp = _get_qdp()
        QdpEngine = getattr(qdp, "QdpEngine", None)
        if QdpEngine is None:
            raise RuntimeError("_qdp.QdpEngine not found. Build with maturin develop.")
        engine = QdpEngine(device_id=self._device_id)
        X = self._array
        if X is None:
            raise RuntimeError(
                "Internal error: _array_iterator called without source_array() data."
            )
        assert X is not None  # type narrowing for static checkers
        n = X.shape[0]
        for start in range(0, n, self._batch_size):
            end = min(start + self._batch_size, n)
            qt = engine.encode(X[start:end], self._num_qubits, self._encoding_method)
            yield qt

    def _create_iterator(self) -> Iterator[object]:
        """Build engine and return the Rust-backed loader iterator (synthetic or file) or array iterator."""
        if self._array_requested:
            if self._synthetic_requested or self._file_requested:
                raise ValueError(
                    "Cannot combine source_array() with source_synthetic() or source_file(); use only one source."
                )
            if self._array is None:
                raise ValueError(
                    "source_array() was called without an array; set with .source_array(X)."
                )
            qdp = _get_qdp()
            engine = getattr(qdp, "QdpEngine", None)
            if engine is None:
                raise RuntimeError(
                    "_qdp.QdpEngine not found. Build with maturin develop."
                )
            engine = engine(device_id=self._device_id)
            create_array_loader = getattr(engine, "create_array_loader", None)
            if create_array_loader is not None:
                return iter(
                    create_array_loader(
                        self._array,
                        batch_size=self._batch_size,
                        num_qubits=self._num_qubits,
                        encoding_method=self._encoding_method,
                        batch_limit=None,
                    )
                )
            return iter(self._array_iterator())
        if self._synthetic_requested and self._file_requested:
            raise ValueError(
                "Cannot set both synthetic and file sources; use either .source_synthetic() or .source_file(path), not both."
            )
        if self._file_requested and self._file_path is None:
            raise ValueError(
                "source_file() was not called with a path; set file source with .source_file(path)."
            )
        use_synthetic = not self._file_requested
        if use_synthetic:
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
        if not use_synthetic:
            if self._streaming_requested:
                create_loader = getattr(engine, "create_streaming_file_loader", None)
                if create_loader is None:
                    raise RuntimeError(
                        "create_streaming_file_loader not available (e.g. only on Linux with CUDA)."
                    )
            else:
                create_loader = getattr(engine, "create_file_loader", None)
                if create_loader is None:
                    raise RuntimeError(
                        "create_file_loader not available (e.g. only on Linux with CUDA)."
                    )
            return iter(
                create_loader(
                    path=self._file_path,
                    batch_size=self._batch_size,
                    num_qubits=self._num_qubits,
                    encoding_method=self._encoding_method,
                    batch_limit=None,
                )
            )
        create_synthetic_loader = getattr(engine, "create_synthetic_loader", None)
        if create_synthetic_loader is None:
            raise RuntimeError(
                "create_synthetic_loader not available (e.g. only on Linux with CUDA)."
            )
        return iter(
            create_synthetic_loader(
                total_batches=self._total_batches,
                batch_size=self._batch_size,
                num_qubits=self._num_qubits,
                encoding_method=self._encoding_method,
                seed=self._seed,
            )
        )

    def _wrap_iterator(self, raw_iter: Iterator[object]) -> Iterator[Any]:
        if self._output_format is None:
            yield from raw_iter
            return
        kind = self._output_format[0]
        if kind == "torch":
            device = self._output_format[1]
            for qt in raw_iter:
                t = _torch.from_dlpack(qt)
                yield t.cpu() if device == "cpu" else t
        elif kind == "numpy":
            for qt in raw_iter:
                yield _torch.from_dlpack(qt).cpu().numpy()
        else:
            yield from raw_iter

    def __iter__(self) -> Iterator[object]:
        """Return iterator yielding one batch per iteration (DLPack, torch, or numpy per as_torch/as_numpy)."""
        raw = self._create_iterator()
        if self._output_format is None:
            return raw
        return self._wrap_iterator(raw)
