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
import os
import sys
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING

from qumat_qdp._backend import get_qdp as _get_qdp

if TYPE_CHECKING:
    import _qdp

# Seed must fit Rust u64: 0 <= seed <= 2^64 - 1.
_U64_MAX = 2**64 - 1

# Accepted dtype aliases for .dtype(); forwarded verbatim to the native loader,
# which parses them case-insensitively (Dtype::from_str_ci).
_VALID_DTYPES = frozenset({"float32", "f32", "float64", "f64"})

# Canonical encoding names (must match Encoding enum in qdp-core/src/types.rs).
_VALID_ENCODINGS: frozenset[str] = frozenset(
    {"amplitude", "angle", "basis", "iqp", "iqp-z", "phase"}
)

# Fallback-supported file extensions (loadable without _qdp).
_TORCH_FILE_EXTS = frozenset({".pt", ".pth"})
_NUMPY_FILE_EXTS = frozenset({".npy"})
_FALLBACK_FILE_EXTS = _TORCH_FILE_EXTS | _NUMPY_FILE_EXTS

# Streaming (Rust-backed) only supports columnar formats.
_STREAMING_FILE_EXTS = frozenset({".parquet"})

# All file extensions accepted as `.source_file()` inputs (Rust + fallback).
_ARROW_FILE_EXTS = frozenset({".arrow", ".feather", ".ipc"})
_PROTOBUF_FILE_EXTS = frozenset({".pb"})
_SUPPORTED_FILE_EXTS = (
    _STREAMING_FILE_EXTS
    | _ARROW_FILE_EXTS
    | _NUMPY_FILE_EXTS
    | _TORCH_FILE_EXTS
    | _PROTOBUF_FILE_EXTS
)

# Backend selection literals.
_BACKEND_RUST = "rust"
_BACKEND_PYTORCH = "pytorch"
_BACKEND_AUTO = "auto"
_VALID_BACKENDS = frozenset({_BACKEND_RUST, _BACKEND_PYTORCH, _BACKEND_AUTO})


def _select_torch_device(torch, device_id: int) -> str:
    """Pick a torch device the current PyTorch build can actually use.

    ``torch.cuda.is_available()`` returns True whenever a usable driver and at
    least one GPU are present, but does not check whether the GPU's compute
    capability is in the PyTorch wheel's compiled arch list. Running on an
    unsupported GPU surfaces as ``cudaErrorNoKernelImageForDevice`` the first
    time a kernel launches -- a particularly opaque failure for users on
    Pascal-and-earlier hardware where recent PyTorch wheels no longer ship
    matching kernels.

    Intersect the device's capability with ``torch.cuda.get_arch_list()`` and
    fall back to CPU (with a warning) when they don't match. Raises
    ``ValueError`` on an out-of-range ``device_id`` to preserve the prior
    contract for callers that explicitly request a specific GPU.
    """
    if not torch.cuda.is_available():
        return "cpu"

    if device_id < 0 or device_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid CUDA device_id {device_id}; "
            f"{torch.cuda.device_count()} device(s) available."
        )

    arch_list = torch.cuda.get_arch_list()
    if arch_list:
        major, minor = torch.cuda.get_device_capability(device_id)
        device_arch = f"sm_{major}{minor}"
        if device_arch not in arch_list:
            warnings.warn(
                f"GPU {device_id} ({torch.cuda.get_device_name(device_id)}, "
                f"{device_arch}) is not in this PyTorch build's supported "
                f"arch list ({sorted(arch_list)}). Falling back to CPU. "
                "Install a PyTorch wheel that targets this GPU, or set "
                "CUDA_VISIBLE_DEVICES= to silence this warning.",
                stacklevel=2,
            )
            return "cpu"

    return f"cuda:{device_id}"


def _path_extension(path: str) -> str:
    """Return the lowercase extension of `path` (handling remote URLs/queries)."""
    is_remote = "://" in path
    tail = path.split("?", 1)[0].rsplit("/", 1)[-1] if is_remote else path
    return os.path.splitext(tail)[1].lower()


def _platform_hint(reason: str) -> str:
    """Return a user-facing hint when the Rust extension is unavailable on a
    non-Linux platform; empty string on Linux."""
    if sys.platform == "linux":
        return ""
    return f" Note: {reason} requires Linux with CUDA; you are on {sys.platform}."


# Cached IterableDataset subclass — built on first call to `as_torch_dataset()` so
# import-time cost is paid only when the user actually opts into the torch adapter.
_torch_dataset_cls: type | None = None


def _build_torch_dataset(loader: QuantumDataLoader):
    """Return a ``torch.utils.data.IterableDataset`` wrapping ``loader``.

    The dataset class is defined once (on first call) and reused thereafter.
    """
    global _torch_dataset_cls
    if _torch_dataset_cls is None:
        try:
            import torch  # noqa: F401 — verifies torch is importable
            from torch.utils.data import IterableDataset
        except ImportError:
            raise RuntimeError(
                "as_torch_dataset() requires PyTorch. Install with: pip install torch"
            ) from None

        class _QdpDataset(IterableDataset):  # type: ignore[misc]
            """IterableDataset wrapping a QuantumDataLoader."""

            def __init__(self, source: QuantumDataLoader) -> None:
                super().__init__()
                self._source = source

            def __iter__(self) -> Iterator[object]:
                import torch as _torch

                for batch in self._source:
                    # DLPack capsule from the Rust backend -> torch.Tensor
                    if not isinstance(batch, _torch.Tensor):
                        batch = _torch.from_dlpack(batch)
                    yield batch

        _torch_dataset_cls = _QdpDataset
    return _torch_dataset_cls(loader)


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
    if encoding_method.lower() not in _VALID_ENCODINGS:
        raise ValueError(
            f"Unknown encoding_method {encoding_method!r}. "
            f"Valid options: {sorted(_VALID_ENCODINGS)}"
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
    """Build a single deterministic sample vector for the given encoding method.

    Supports amplitude, angle, basis, iqp, and iqp-z.
    """
    import numpy as np

    if encoding_method == "basis":
        mask = np.uint64(vector_len - 1)
        idx = np.uint64(seed) & mask
        return [float(idx)]
    if encoding_method in ("angle", "iqp", "iqp-z"):
        if vector_len == 0:
            return []
        scale = (2.0 * math.pi) / vector_len
        idx = np.arange(vector_len, dtype=np.uint64)
        mixed = (idx + np.uint64(seed)) % np.uint64(vector_len)
        return (mixed.astype(np.float64) * scale).tolist()
    # amplitude
    mask = np.uint64(vector_len - 1)
    scale = 1.0 / vector_len
    idx = np.arange(vector_len, dtype=np.uint64)
    mixed = (idx + np.uint64(seed)) & mask
    return (mixed.astype(np.float64) * scale).tolist()


def _sample_dim(num_qubits: int, encoding_method: str) -> int:
    """Return the synthetic sample dimension for the selected encoding."""
    if encoding_method == "basis":
        return 1
    if encoding_method in ("angle", "iqp-z"):
        return num_qubits
    if encoding_method == "iqp":
        return num_qubits + num_qubits * (num_qubits - 1) // 2
    return 1 << num_qubits


class QuantumDataLoader:
    """
    Builder for batched QDP encoding iterators.

    ``QuantumDataLoader`` can generate synthetic input samples or read supported
    file formats, then encode each batch with the selected backend.  The default
    ``"rust"`` backend returns Rust-backed ``QuantumTensor`` batches, while the
    explicit ``"pytorch"`` backend returns ``torch.Tensor`` batches.  The
    ``"auto"`` backend tries the Rust extension first and falls back to PyTorch
    when the native extension is unavailable.
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
        """Create a loader builder with default synthetic batching settings.

        :param device_id: GPU device ordinal used by native and PyTorch backends.
        :param num_qubits: Number of qubits in each encoded output state.
        :param batch_size: Number of samples per emitted batch.
        :param total_batches: Maximum number of batches to emit.
        :param encoding_method: Encoding method name.
        :param seed: Optional synthetic data seed.
        :raises ValueError: If any initial setting is invalid.
        """
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
        self._dtype: str | None = None  # None -> native default (float64, lossless)
        self._backend_name: str = _BACKEND_RUST

    def qubits(self, n: int) -> QuantumDataLoader:
        """Set the number of qubits used by subsequent encodings.

        ``n`` must be a positive integer.  The value controls the encoded state
        size (for example, amplitude and phase-style encodings produce vectors
        of length ``2**n``) and the expected input width for encodings such as
        ``"angle"`` and ``"iqp-z"``.

        :param n: Positive qubit count.
        :returns: ``self`` for fluent builder chaining.
        :raises ValueError: If ``n`` is not a positive integer.
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"num_qubits must be a positive integer, got {n!r}")
        self._num_qubits = n
        return self

    def encoding(self, method: str) -> QuantumDataLoader:
        """Set the quantum feature encoding method.

        Valid values are ``"amplitude"``, ``"angle"``, ``"basis"``,
        ``"iqp"``, ``"iqp-z"``, and ``"phase"``.  Use these canonical
        lowercase names because the selected backend receives the string exactly
        as supplied.  The PyTorch reference backend supports the same methods as
        :mod:`qumat_qdp.torch_ref`; use the native backend for methods that are
        not available in the reference path.

        :param method: Encoding method name.
        :returns: ``self`` for fluent builder chaining.
        :raises ValueError: If ``method`` is empty, not a string, or not a
            supported encoding.
        """
        if not method or not isinstance(method, str):
            raise ValueError(
                f"encoding_method must be a non-empty string, got {method!r}"
            )
        if method.lower() not in _VALID_ENCODINGS:
            raise ValueError(
                f"Unknown encoding {method!r}. "
                f"Valid options: {sorted(_VALID_ENCODINGS)}"
            )
        self._encoding_method = method
        return self

    def batches(self, total: int, size: int = 64) -> QuantumDataLoader:
        """Set the number of batches to produce and samples per batch.

        Both ``total`` and ``size`` must be positive integers.  For synthetic
        sources, ``total`` is the exact number of generated batches.  For file
        sources handled by the PyTorch fallback, iteration stops at the smaller
        of ``total`` and the number of complete/partial batches available from
        the loaded file.

        :param total: Positive maximum number of batches to emit.
        :param size: Positive number of samples per encoded batch.
        :returns: ``self`` for fluent builder chaining.
        :raises ValueError: If either argument is not a positive integer.
        """
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
        """Select the synthetic data source.

        Synthetic data is the default when no file source is configured, but
        calling this method records the source choice explicitly.  Use
        ``seed()`` to make generated samples reproducible where the selected
        backend supports seeded generation.  If ``total_batches`` is provided,
        it overrides the current batch count and must be a positive integer.
        Selecting both ``source_synthetic()`` and ``source_file()`` on the same
        loader is rejected when iteration starts.

        :param total_batches: Optional positive replacement for the configured
            number of batches.
        :returns: ``self`` for fluent builder chaining.
        :raises ValueError: If ``total_batches`` is provided but is not a
            positive integer.
        """
        self._synthetic_requested = True
        if total_batches is not None:
            if not isinstance(total_batches, int) or total_batches < 1:
                raise ValueError(
                    f"total_batches must be a positive integer, got {total_batches!r}"
                )
            self._total_batches = total_batches
        return self

    def source_file(self, path: str, streaming: bool = False) -> QuantumDataLoader:
        """Use a file data source.

        Non-streaming native loading accepts ``.parquet``, ``.arrow``,
        ``.feather``, ``.ipc``, ``.npy``, ``.pt``, ``.pth``, and ``.pb`` files.
        The PyTorch fallback path supports only ``.npy``, ``.pt``, and ``.pth``
        inputs because it loads the full tensor into memory before encoding.
        Streaming mode is native-only and currently accepts ``.parquet`` files.
        Remote ``s3://`` and ``gs://`` paths are accepted when the native remote
        I/O feature is enabled; remote query strings and fragments are rejected.

        Element precision is controlled by :meth:`dtype`. By default file input is
        loaded as ``float64`` (lossless). Selecting ``dtype("float32")`` narrows f64
        file contents to f32 on load; values outside the f32 range become ``±Inf``.
        ``basis`` encoding is exempt: its values are integer state indices, so it is
        always loaded as f64 and an explicit ``float32`` request is rejected when an
        index exceeds f32's exact integer range (``2**24``).

        :param path: Local or supported remote input path.
        :param streaming: Whether to request native streaming file loading.
        :returns: ``self`` for fluent builder chaining.
        :raises ValueError: If ``path`` is empty, includes an unsupported remote
            query/fragment, or requests streaming for an unsupported extension.
        """
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path!r}")
        if "://" in path and ("?" in path or "#" in path):
            raise ValueError(
                "Remote URL query/fragment is not supported; use plain scheme://bucket/key path."
            )

        # Reject streaming=True for unsupported formats at builder time (not iteration
        # time) so the error is immediate and actionable before any data is read.
        if streaming and _path_extension(path) not in _STREAMING_FILE_EXTS:
            raise ValueError(
                f"streaming=True supports only {sorted(_STREAMING_FILE_EXTS)} files; "
                "use streaming=False for other formats."
            )
        self._file_path = path
        self._file_requested = True
        self._streaming_requested = streaming
        return self

    def dtype(self, name: str) -> QuantumDataLoader:
        """Set the element precision used when loading file sources.

        Applies to the native :meth:`source_file` path. ``"float64"`` (the default)
        loads file contents losslessly; ``"float32"`` narrows them to f32 on load.
        See :meth:`source_file` for the cast caveats, including the ``basis``
        exemption.

        :param name: ``"float32"``/``"f32"`` or ``"float64"``/``"f64"`` (case-insensitive).
        :returns: ``self`` for fluent builder chaining.
        :raises ValueError: If ``name`` is not a recognized dtype.
        """
        if not isinstance(name, str) or name.strip().lower() not in _VALID_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(_VALID_DTYPES)}, got {name!r}"
            )
        self._dtype = name.strip().lower()
        return self

    def seed(self, s: int | None = None) -> QuantumDataLoader:
        """Set or clear the synthetic data seed.

        ``None`` leaves the loader unseeded for the native Rust path and maps to
        the PyTorch reference path's default deterministic seed.  Integer seeds
        must fit Rust ``u64`` so the same configuration can be passed to the
        native backend.

        :param s: ``None`` or an integer in ``[0, 2**64 - 1]``.
        :returns: ``self`` for fluent builder chaining.
        :raises ValueError: If ``s`` is not ``None`` or a valid Rust ``u64``.
        """
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
        """Set how nullable file inputs are handled by the native loader.

        Valid policies are ``"fill_zero"`` (replace nulls with zero before
        encoding) and ``"reject"`` (fail on null input).  The policy is passed
        through to Rust file and synthetic loader creation when available.  The
        PyTorch fallback loaders do not consume this setting because supported
        ``.npy``/``.pt``/``.pth`` inputs are loaded as dense tensors.

        :param policy: Null handling policy, either ``"fill_zero"`` or
            ``"reject"``.
        :returns: ``self`` for fluent builder chaining.
        :raises ValueError: If ``policy`` is not supported.
        """
        if policy not in ("fill_zero", "reject"):
            raise ValueError(
                f"null_handling must be 'fill_zero' or 'reject', got {policy!r}"
            )
        self._null_handling = policy
        return self

    def backend(self, name: str) -> QuantumDataLoader:
        """Set encoding backend: ``'rust'``, ``'pytorch'``, or ``'auto'``.

        ``'auto'``: tries the Rust backend first and falls back to the PyTorch
        reference backend if the Rust extension is unavailable, emitting a
        ``RuntimeWarning`` when the fallback occurs.  ``'rust'`` raises if the
        extension is missing.  ``'pytorch'`` always uses the pure-PyTorch path.
        Returns self for chaining.
        """
        if name not in _VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(_VALID_BACKENDS)}, got {name!r}"
            )
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
        if (
            not use_synthetic
            and self._file_path is not None
            and self._backend_name != _BACKEND_PYTORCH
        ):
            ext = _path_extension(self._file_path)
            if ext not in _SUPPORTED_FILE_EXTS:
                raise ValueError(
                    f"Unsupported file extension {ext!r}. "
                    f"Supported: {', '.join(sorted(_SUPPORTED_FILE_EXTS))}"
                )
            is_remote = "://" in self._file_path
            if not is_remote and not os.path.exists(self._file_path):
                raise FileNotFoundError(
                    f"File not found: {self._file_path!r}. Check the path and try again."
                )
        if use_synthetic:
            _validate_loader_args(
                device_id=self._device_id,
                num_qubits=self._num_qubits,
                batch_size=self._batch_size,
                total_batches=self._total_batches,
                encoding_method=self._encoding_method,
                seed=self._seed,
            )
        if self._backend_name == _BACKEND_PYTORCH:
            return self._create_pytorch_iterator(use_synthetic)
        # Rust backend (default) or auto-fallback.
        qdp = _get_qdp()
        QdpEngine = getattr(qdp, "QdpEngine", None) if qdp else None
        if QdpEngine is None:
            if self._backend_name == _BACKEND_AUTO:
                warnings.warn(
                    "Rust extension (_qdp) is not available"
                    f"{_platform_hint('the Rust GPU extension')}; "
                    "falling back to PyTorch reference backend. "
                    "Build with: maturin develop to enable GPU acceleration.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return self._create_pytorch_iterator(use_synthetic)
            raise RuntimeError(
                "Rust extension (_qdp) is not available. "
                "Build with: maturin develop, or explicitly select the PyTorch "
                f"reference backend with .backend({_BACKEND_PYTORCH!r})."
                f"{_platform_hint('the Rust GPU extension')}"
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
                    dtype=self._dtype,
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

        device = _select_torch_device(torch, self._device_id)

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
        sample_size = _sample_dim(num_qubits, encoding_method)

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
        path = self._file_path
        assert path is not None
        ext = _path_extension(path)

        if self._streaming_requested:
            raise RuntimeError(
                "Streaming file loading requires the _qdp Rust extension. "
                "Build with: maturin develop"
                f"{_platform_hint('streaming')}"
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

    def as_torch_dataset(self):
        """Wrap this loader as a ``torch.utils.data.IterableDataset``.

        Returns a dataset that yields one encoded batch (``torch.Tensor``) per
        iteration step, compatible with ``torch.utils.data.DataLoader``.

        Example::

            from qumat_qdp import QuantumDataLoader
            import torch

            dataset = (QuantumDataLoader()
                       .qubits(16).encoding("amplitude")
                       .batches(100, size=64)
                       .source_synthetic()
                       .as_torch_dataset())
            loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
            for batch in loader:
                ...  # batch is torch.Tensor, shape (64, 2**16)

        Note: ``batch_size=None`` in DataLoader disables DataLoader's own batching;
        ``num_workers=0`` is required because the Rust backend holds GPU state that
        cannot be pickled for multi-process workers.
        """
        return _build_torch_dataset(self)

    def __iter__(self) -> Iterator[object]:
        """Return iterator that yields one encoded batch per step.

        With the default ``"rust"`` backend, yields ``QuantumTensor``
        (use ``torch.from_dlpack(qt)``).  With ``.backend("pytorch")``,
        yields ``torch.Tensor`` directly.
        """
        return self._create_iterator()
