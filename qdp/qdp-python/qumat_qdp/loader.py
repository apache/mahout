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

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

from qumat_qdp._backend import get_qdp as _get_qdp

if TYPE_CHECKING:
    import _qdp

# Seed must fit Rust u64: 0 <= seed <= 2^64 - 1.
_U64_MAX = 2**64 - 1

# Fallback-supported file extensions (loadable without _qdp).
_TORCH_FILE_EXTS = frozenset({".pt", ".pth"})
_NUMPY_FILE_EXTS = frozenset({".npy"})
_FALLBACK_FILE_EXTS = _TORCH_FILE_EXTS | _NUMPY_FILE_EXTS


def _validate_loader_args(
    *,
    device_id: int,
    num_qubits: int,
    batch_size: int,
    total_batches: int,
    encoding_method: str,
    seed: int | None,
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


def _build_sample(seed: int, vector_len: int, encoding_method: str) -> list[float]:
    """Build a single deterministic sample vector (mirrors benchmark/utils.py:build_sample)."""
    import numpy as np

    if encoding_method == "basis":
        mask = np.uint64(vector_len - 1)
        idx = np.uint64(seed) & mask
        return [float(idx)]
    if encoding_method == "angle":
        if vector_len == 0:
            return []
        scale = (2.0 * math.pi) / vector_len
        idx = np.arange(vector_len, dtype=np.uint64)
        mixed = (idx + np.uint64(seed)) % np.uint64(vector_len)
        return (mixed.astype(np.float64) * scale).tolist()
    # amplitude / iqp
    mask = np.uint64(vector_len - 1)
    scale = 1.0 / vector_len
    idx = np.arange(vector_len, dtype=np.uint64)
    mixed = (idx + np.uint64(seed)) & mask
    return (mixed.astype(np.float64) * scale).tolist()


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
        seed: int | None = None,
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
        self._file_path: str | None = None
        self._streaming_requested = (
            False  # set True by source_file(streaming=True); Phase 2b
        )
        self._synthetic_requested = False  # set True only by source_synthetic()
        self._file_requested = False
        self._null_handling: str | None = None
        self._backend_name: str = "rust"

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
        total_batches: int | None = None,
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
        Remote paths (s3://, gs://) are supported when the remote-io feature is enabled.
        Remote URL query/fragment (for example ?versionId=... or #...) is not supported.
        """
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path!r}")
        if "://" in path and ("?" in path or "#" in path):
            raise ValueError(
                "Remote URL query/fragment is not supported; use plain scheme://bucket/key path."
            )
        # For remote URLs, extract the key portion for extension checks.
        check_path = path.split("?")[0].rsplit("/", 1)[-1] if "://" in path else path
        if streaming and not (check_path.lower().endswith(".parquet")):
            raise ValueError(
                "streaming=True supports only .parquet files; use streaming=False for other formats."
            )
        self._file_path = path
        self._file_requested = True
        self._streaming_requested = streaming
        return self

    def seed(self, s: int | None = None) -> QuantumDataLoader:
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

    def null_handling(self, policy: str) -> QuantumDataLoader:
        """Set null handling policy ('fill_zero' or 'reject'). Returns self for chaining."""
        if policy not in ("fill_zero", "reject"):
            raise ValueError(
                f"null_handling must be 'fill_zero' or 'reject', got {policy!r}"
            )
        self._null_handling = policy
        return self

    def backend(self, name: str) -> QuantumDataLoader:
        """Set encoding backend: ``'rust'`` or ``'pytorch'``.

        The PyTorch reference backend is intended for testing and must be
        explicitly selected.  Returns self for chaining.
        """
        if name not in ("rust", "pytorch"):
            raise ValueError(f"backend must be 'rust' or 'pytorch', got {name!r}")
        self._backend_name = name
        return self

    def _create_iterator(self) -> Iterator[object]:
        """Build engine and return a loader iterator (Rust-backed or PyTorch fallback)."""
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
        if self._backend_name == "pytorch":
            return self._create_pytorch_iterator(use_synthetic)
        # Rust backend (default).
        qdp = _get_qdp()
        QdpEngine = getattr(qdp, "QdpEngine", None) if qdp else None
        if QdpEngine is None:
            raise RuntimeError(
                "Rust extension (_qdp) is not available. "
                "Build with: maturin develop, or explicitly select the PyTorch "
                "reference backend with .backend('pytorch')."
            )
        return self._create_rust_iterator(QdpEngine, use_synthetic)

    def _create_rust_iterator(self, QdpEngine, use_synthetic: bool) -> Iterator[object]:
        """Create the Rust-backed loader iterator (original path)."""
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
                    null_handling=self._null_handling,
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
                null_handling=self._null_handling,
            )
        )

    def _create_pytorch_iterator(self, use_synthetic: bool) -> Iterator[object]:
        """PyTorch reference iterator (explicitly selected via ``.backend('pytorch')``).

        Yields ``torch.Tensor`` (not ``QuantumTensor``).

        Supports synthetic data and ``.npy`` / ``.pt`` / ``.pth`` files.
        Parquet, Arrow, and streaming sources require the Rust extension.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch backend selected but torch is not installed. "
                "Install PyTorch with: pip install torch"
            ) from None

        from qumat_qdp.torch_ref import encode

        device = f"cuda:{self._device_id}" if torch.cuda.is_available() else "cpu"

        if use_synthetic:
            return self._pytorch_synthetic_iter(torch, encode, device)
        return self._pytorch_file_iter(torch, encode, device)

    def _pytorch_synthetic_iter(
        self, torch, encode_fn, device: str
    ) -> Iterator[object]:
        """Generate synthetic data and encode with PyTorch."""
        import numpy as np

        num_qubits = self._num_qubits
        encoding_method = self._encoding_method
        batch_size = self._batch_size
        seed = self._seed if self._seed is not None else 0

        if encoding_method == "basis":
            sample_size = 1
        elif encoding_method == "angle":
            sample_size = num_qubits
        elif encoding_method == "iqp":
            sample_size = num_qubits + num_qubits * (num_qubits - 1) // 2
        else:
            sample_size = 1 << num_qubits

        for batch_idx in range(self._total_batches):
            base = batch_idx * batch_size
            samples = []
            for i in range(batch_size):
                samples.append(
                    _build_sample(base + i + seed, sample_size, encoding_method)
                )
            batch_np = np.stack(samples)
            batch_tensor = torch.tensor(batch_np, dtype=torch.float64, device=device)
            yield encode_fn(batch_tensor, num_qubits, encoding_method, device=device)

    def _pytorch_file_iter(self, torch, encode_fn, device: str) -> Iterator[object]:
        """Load file data and encode with PyTorch."""
        import os

        path = self._file_path
        assert path is not None
        ext = os.path.splitext(path)[1].lower()

        if self._streaming_requested:
            raise RuntimeError(
                "Streaming file loading requires the _qdp Rust extension. "
                "Build with: maturin develop"
            )

        if ext not in _FALLBACK_FILE_EXTS:
            raise RuntimeError(
                f"PyTorch fallback only supports {', '.join(sorted(_FALLBACK_FILE_EXTS))} files. "
                f"Got {ext!r}. Build the _qdp extension for full format support: maturin develop"
            )

        # Load all data into memory.
        if ext in _TORCH_FILE_EXTS:
            raw = torch.load(path, weights_only=True)
            if not isinstance(raw, torch.Tensor):
                raise RuntimeError(
                    f"Expected torch.Tensor in {path}, got {type(raw).__name__}"
                )
            all_data = raw.to(dtype=torch.float64, device=device)
        else:
            import numpy as np

            arr = np.load(path)
            all_data = torch.tensor(arr, dtype=torch.float64, device=device)

        if all_data.ndim == 1:
            all_data = all_data.unsqueeze(0)

        num_qubits = self._num_qubits
        encoding_method = self._encoding_method
        batch_size = self._batch_size
        total_samples = all_data.shape[0]
        total_batches = min(
            self._total_batches, (total_samples + batch_size - 1) // batch_size
        )

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_samples)
            batch = all_data[start:end]
            yield encode_fn(batch, num_qubits, encoding_method, device=device)

    def __iter__(self) -> Iterator[object]:
        """Return iterator that yields one encoded batch per step.

        With the default ``"rust"`` backend, yields ``QuantumTensor``
        (use ``torch.from_dlpack(qt)``).  With ``.backend("pytorch")``,
        yields ``torch.Tensor`` directly.
        """
        return self._create_iterator()
