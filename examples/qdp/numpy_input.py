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

"""Feed NumPy data to QDP: in-memory arrays and .npy files.

See docs/qdp/numpy-input.md for the full rules. The preparation steps run
anywhere; the encode steps need the _qdp extension and a CUDA GPU and are
skipped otherwise.

    python examples/qdp/numpy_input.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def prepare(raw: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float64 2D array, which QDP accepts as-is."""
    x = np.ascontiguousarray(raw, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"expected 1D or 2D data, got {x.ndim}D")
    return x


def main() -> None:
    rng = np.random.default_rng(0)

    # Typical "wrong" inputs: float32 and a transposed (non-contiguous) view.
    features_f32 = rng.standard_normal((16, 8)).astype(np.float32)
    transposed = rng.standard_normal((8, 16)).T
    for name, raw in [("float32", features_f32), ("transposed", transposed)]:
        x = prepare(raw)
        print(
            f"{name:>10}: dtype {raw.dtype} -> {x.dtype}, "
            f"C-contiguous {raw.flags['C_CONTIGUOUS']} -> {x.flags['C_CONTIGUOUS']}, "
            f"shape {x.shape}"
        )

    batch = prepare(features_f32)  # 16 samples x 8 features -> 3 qubits
    num_qubits = 3

    with tempfile.TemporaryDirectory() as tmp:
        npy_path = Path(tmp) / "features.npy"
        np.save(npy_path, batch)  # 2D float64: what the .npy reader expects
        print(f"saved {npy_path.name}: shape {batch.shape}, dtype {batch.dtype}")

        try:
            from qumat_qdp import QdpEngine, is_cuda_available
        except ImportError:
            print("qumat_qdp not installed; skipping encode")
            return
        if not is_cuda_available():
            print("no CUDA device (or a CUDA-stub build); skipping encode")
            return

        import torch

        engine = QdpEngine(device_id=0, precision="float64")
        from_array = torch.from_dlpack(engine.encode(batch, num_qubits))
        from_file = torch.from_dlpack(engine.encode(npy_path, num_qubits))
        print(f"encoded states: {tuple(from_array.shape)} {from_array.dtype}")
        print(f"array and file results match: {torch.allclose(from_array, from_file)}")

        # Amplitude encoding stores x / ||x||, so each state has unit norm.
        norms = from_array.abs().pow(2).sum(dim=1).sqrt()
        print(f"max |norm - 1|: {float((norms - 1).abs().max()):.2e}")


if __name__ == "__main__":
    main()
