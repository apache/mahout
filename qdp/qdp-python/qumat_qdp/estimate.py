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

"""Upfront memory estimation for QDP pipeline configurations.

``QuantumDataLoader`` rejects a configuration whose device buffers cannot fit in
free VRAM when iteration starts.  This module answers the same question *before*
building anything, so a caller can size ``num_qubits``/``batch_size`` against a
memory budget rather than discovering the ceiling from a rejection.

Usage::

    from qumat_qdp import estimate_memory

    est = estimate_memory(num_qubits=20, batch_size=64, dtype="f32")
    print(est.gpu_state_bytes / 1024**2, "MiB of device state")
"""

from __future__ import annotations

from dataclasses import dataclass

from qumat_qdp._backend import get_qdp

_NO_EXTENSION = (
    "Memory estimation requires the native QDP extension (_qdp), which is not "
    "available. Build it with: uv run --active maturin develop --manifest-path "
    "qdp/qdp-python/Cargo.toml"
)


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated memory footprint of a pipeline configuration.

    Returned by :func:`estimate_memory`.  Every field is an upper-bound estimate
    derived from configuration arithmetic alone -- nothing here reflects an actual
    allocation, and no device is consulted.
    """

    cpu_prefetch_bytes: int
    """Host prefetch pool: ``prefetch_depth`` batches of raw, unencoded input."""

    gpu_state_bytes: int
    """Device state-vector buffer, including a double-buffering allowance."""

    total_bytes: int
    """Combined host and device footprint."""


def estimate_memory(
    num_qubits: int,
    batch_size: int,
    encoding_method: str = "amplitude",
    dtype: str = "f64",
    prefetch_depth: int = 16,
) -> MemoryEstimate:
    """Estimate the host and device memory a pipeline configuration would need.

    ``gpu_state_bytes`` is the figure the loader's guard compares against free VRAM
    when iteration starts.  The guard budgets the wider of ``dtype`` and the engine's
    precision, and always f64 for ``"basis"`` read from a file (whose inputs are
    integer state indices read as f64; synthetic basis data is budgeted at the
    requested ``dtype``), so pass the engine precision as ``dtype`` when the two
    differ -- otherwise this estimate is half the one the guard applies.

    :param num_qubits: Qubit count; the device state vector holds ``2**num_qubits``
        complex amplitudes per sample.
    :param batch_size: Samples per batch.
    :param encoding_method: ``"amplitude"``, ``"angle"``, ``"basis"``, ``"iqp"``,
        ``"iqp-z"``, or ``"phase"`` (case-insensitive).
    :param dtype: ``"float32"``/``"f32"`` or ``"float64"``/``"f64"``
        (case-insensitive).  Encodings without an f32 batch path are estimated as
        f64 regardless, mirroring what the pipeline does with the same request.
    :param prefetch_depth: Host prefetch queue depth.  Affects
        ``cpu_prefetch_bytes`` only; device buffers do not scale with it.
    :returns: The :class:`MemoryEstimate` for this configuration.
    :raises ValueError: If a name is unrecognized, ``2**num_qubits`` is not
        representable, or the arithmetic overflows.
    :raises OverflowError: If ``num_qubits`` or ``batch_size`` is negative, which
        fails at the argument boundary before the estimator runs.
    :raises RuntimeError: If the native extension is not available.
    """
    qdp = get_qdp()
    native = getattr(qdp, "estimate_memory", None) if qdp is not None else None
    if native is None:
        raise RuntimeError(_NO_EXTENSION)

    cpu_prefetch_bytes, gpu_state_bytes, total_bytes = native(
        num_qubits=num_qubits,
        batch_size=batch_size,
        encoding_method=encoding_method,
        dtype=dtype,
        prefetch_depth=prefetch_depth,
    )
    return MemoryEstimate(
        cpu_prefetch_bytes=cpu_prefetch_bytes,
        gpu_state_bytes=gpu_state_bytes,
        total_bytes=total_bytes,
    )
