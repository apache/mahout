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

"""CPU-side input validation and L2 norms used before QDP encoding.

``Preprocessor`` exposes the same checks the native encoders run before any
data is copied to the GPU, so callers can validate inputs (and inspect the
norm amplitude encoding will divide by) without a GPU. The work is done by
the Rust ``_qdp`` extension; this module only adds documentation and a clear
error when the extension is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qumat_qdp._backend import get_qdp

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike

_MISSING_EXTENSION_MSG = (
    "qumat_qdp.Preprocessor requires the _qdp Rust extension. "
    "Build it with: maturin develop --manifest-path qdp/qdp-python/Cargo.toml"
)


def _native() -> Any:
    qdp = get_qdp()
    native = getattr(qdp, "Preprocessor", None) if qdp is not None else None
    if native is None:
        raise RuntimeError(_MISSING_EXTENSION_MSG)
    return native


class Preprocessor:
    """Validate classical input and compute L2 norms before encoding.

    All methods are static and run on the CPU; no GPU or CUDA toolkit is
    needed. Inputs may be NumPy arrays or plain Python sequences. Values are
    converted to ``float64`` (a C-contiguous ``float64`` array is used without
    a copy). Every method raises ``RuntimeError`` with a message starting
    with ``"Invalid input:"`` when a check fails, matching the errors that
    ``QdpEngine.encode`` reports for the same input.

    Example -- check a sample, then compute the amplitudes that amplitude
    encoding will load::

        import numpy as np
        from qumat_qdp import Preprocessor

        x = np.array([3.0, 4.0])          # 2 features
        Preprocessor.validate_input(x, num_qubits=1)
        norm = Preprocessor.calculate_l2_norm(x)   # 5.0
        amplitudes = x / norm                      # [0.6, 0.8]

    Amplitude encoding stores ``x / ||x||`` so that the squared amplitudes
    (measurement probabilities) sum to 1. A zero vector has no such
    normalization, so it is rejected rather than silently encoded.

    :raises RuntimeError: From any method, if the ``_qdp`` extension is not
        installed.
    """

    @staticmethod
    def validate_input(data: ArrayLike, num_qubits: int) -> None:
        """Check that one sample can be encoded with ``num_qubits`` qubits.

        :param data: One sample as a 1D array or sequence of floats.
        :param num_qubits: Number of qubits in the target state, between 1
            and 30 inclusive.
        :returns: ``None`` when the input is valid.
        :raises RuntimeError: If ``num_qubits`` is 0 or greater than 30, if
            ``data`` is empty, or if ``len(data) > 2**num_qubits``.
        :raises TypeError: If ``data`` is not one-dimensional.
        :raises ValueError: If ``data`` holds values NumPy cannot convert to
            ``float64`` (for example strings or ragged nested lists).
        """
        _native().validate_input(data, num_qubits)

    @staticmethod
    def calculate_l2_norm(data: ArrayLike) -> float:
        """Return the Euclidean (L2) norm ``sqrt(sum(x_i ** 2))`` of a sample.

        This is the value amplitude encoding divides by. The sum is computed
        in parallel, so the last bits may differ from ``numpy.linalg.norm``.

        :param data: One sample as a 1D array or sequence of floats.
        :returns: The norm as a positive, finite Python ``float``.
        :raises RuntimeError: If the norm is zero (including empty input), or
            if ``data`` contains NaN or infinite values.
        :raises TypeError: If ``data`` is not one-dimensional.
        :raises ValueError: If ``data`` holds values NumPy cannot convert to
            ``float64`` (for example strings or ragged nested lists).
        """
        return _native().calculate_l2_norm(data)

    @staticmethod
    def validate_batch(
        batch_data: ArrayLike,
        num_samples: int,
        sample_size: int,
        num_qubits: int,
    ) -> None:
        """Check that a batch of samples can be encoded with ``num_qubits``.

        :param batch_data: Either a 2D array of shape
            ``(num_samples, sample_size)`` or the same values flattened to 1D
            in row-major order.
        :param num_samples: Number of samples in the batch; must be > 0.
        :param sample_size: Number of values in each sample; must be at most
            ``2**num_qubits``.
        :param num_qubits: Number of qubits in each target state, between 1
            and 30 inclusive.
        :returns: ``None`` when the batch is valid.
        :raises RuntimeError: If ``num_samples`` is 0, if the data length is
            not ``num_samples * sample_size`` (or a 2D shape disagrees), if
            ``num_qubits`` is out of range, or if
            ``sample_size > 2**num_qubits``.
        :raises ValueError: If ``batch_data`` holds values NumPy cannot
            convert to ``float64``.
        """
        _native().validate_batch(batch_data, num_samples, sample_size, num_qubits)

    @staticmethod
    def calculate_batch_l2_norms(
        batch_data: ArrayLike,
        num_samples: int,
        sample_size: int,
    ) -> np.ndarray:
        """Return the L2 norm of every sample in a batch.

        :param batch_data: Either a 2D array of shape
            ``(num_samples, sample_size)`` or the same values flattened to 1D
            in row-major order.
        :param num_samples: Number of samples in the batch; must be > 0.
        :param sample_size: Number of values in each sample; must be > 0.
        :returns: A 1D ``float64`` NumPy array of length ``num_samples``.
        :raises RuntimeError: If the data length does not equal
            ``num_samples * sample_size``, or if any sample has a zero, NaN or
            infinite norm (the message names the first bad sample index).
        :raises ValueError: If ``batch_data`` holds values NumPy cannot
            convert to ``float64``.
        """
        return _native().calculate_batch_l2_norms(batch_data, num_samples, sample_size)


__all__ = ["Preprocessor"]
