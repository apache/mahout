#!/usr/bin/env python3
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

"""End-to-end f32 vs f64 Parquet pipeline throughput benchmark (issue #1342).

Reads amplitude data from a Parquet file through ``QuantumDataLoader`` at
``dtype("float32")`` and ``dtype("float64")`` and reports encoded
vectors/second for each, plus the f32/f64 speedup. The expected win is
~25-35% when the Parquet column is native f32 (see #1338).

To actually exercise the native f32 path, the default (no ``--parquet``) run
generates *two* files holding the same logical data — one native
``FixedSizeList<Float32>`` and one native ``FixedSizeList<Float64>`` — and reads
each at its matching dtype, so the f32 run is the zero-copy native-f32 read, not
a f64→f32 cast. With ``--parquet PATH`` the same user file is read at both
dtypes (a native-f32 column hits the zero-copy path; a f64 column is cast to f32
on read).

This is a GPU/Linux-only benchmark: ``QuantumDataLoader``'s native file loader is
only available on Linux with CUDA, mirroring ``benchmark_phase.py``. It is NOT
wired into CI; run it locally on a GPU box.

Run from the repo root::

    uv run --project qdp/qdp-python \\
        python qdp/qdp-python/benchmark/benchmark_parquet_f32.py \\
        --qubits 12 --batches 200 --batch-size 64

The generated data is materialized in RAM and on disk and grows as
``batches * batch_size * 2^qubits``, so raise ``--qubits`` with that cost in mind.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from qumat_qdp import QuantumDataLoader


def _generate_parquet(
    path: str,
    total_rows: int,
    sample_size: int,
    np_dtype: type[np.floating],
    pa_type: pa.DataType,
) -> None:
    """Write a ``FixedSizeList<pa_type, sample_size>`` Parquet file of random data.

    Each call generates its own data from a fixed seed, so the f32 and f64 files
    hold the same logical values without the caller keeping both large host
    arrays alive at once. A FixedSizeList has no offsets buffer, so this also
    sidesteps the int32 offset overflow a variable ``List`` hits once
    ``total_rows * sample_size`` exceeds 2^31 (large --qubits).
    """
    flat = np.random.default_rng(0).random(total_rows * sample_size)
    if np_dtype != np.float64:
        flat = flat.astype(np_dtype)
    values = pa.array(flat, type=pa_type)
    list_array = pa.FixedSizeListArray.from_arrays(values, sample_size)
    pq.write_table(pa.table({"data": list_array}), path)


def run_loader(
    path: str,
    dtype: str,
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    encoding_method: str,
) -> tuple[float, float]:
    """Iterate the loader at the given dtype; return (duration_sec, vec/s).

    The native file loader reads the *entire* file regardless of ``--batches``:
    ``total_batches`` is ignored for file sources (only ``--batch-size`` sets
    batch granularity), and iteration runs until EOF. So the loop processes
    exactly the file's Parquet row count, and ``num_rows / elapsed`` is the true
    throughput for any ``--parquet`` file -- not just the generated case where
    ``num_rows == batches * batch_size``. (Do not clamp to ``batches *
    batch_size``: for a file larger than that, the whole file is still read, so
    clamping would divide the full read time by an undercount.)
    """
    total_vectors = pq.read_metadata(path).num_rows
    loader = (
        QuantumDataLoader(device_id=0)
        .qubits(num_qubits)
        .encoding(encoding_method)
        .batches(total_batches, size=batch_size)
        .dtype(dtype)
        .source_file(path)
    )
    start = time.perf_counter()
    for _ in loader:  # drive the pipeline; the encoded batch is not needed
        pass
    elapsed = max(time.perf_counter() - start, 1e-9)
    return elapsed, total_vectors / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="f32 vs f64 Parquet pipeline throughput benchmark"
    )
    # Default sizes are deliberately modest: the generated data is materialized
    # in host RAM and written to disk (one f32 + one f64 file), so it grows as
    # batches * batch_size * 2^qubits. 12 qubits keeps the default run ~0.4 GB;
    # raise --qubits with that cost in mind.
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument(
        "--batches",
        type=int,
        default=200,
        help="Number of batches. Sizes the generated data (batches * batch_size "
        "rows); ignored with --parquet, where the whole file is always read.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoding", type=str, default="amplitude")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Existing Parquet file to benchmark at both dtypes; "
        "if omitted, native f32 and f64 temp files are generated.",
    )
    args = parser.parse_args()

    sample_size = 1 << args.qubits
    total_rows = args.batches * args.batch_size

    tmp_paths: list[str] = []
    # Map each dtype to the file it should read. For the generated case each
    # dtype reads its own native-precision file (no cast); for --parquet both
    # read the user's single file.
    paths: dict[str, str]
    if args.parquet is None:
        f64_fd, f64_path = tempfile.mkstemp(suffix="_bench_f64.parquet")
        f32_fd, f32_path = tempfile.mkstemp(suffix="_bench_f32.parquet")
        os.close(f64_fd)
        os.close(f32_fd)
        # Register for cleanup before generating, so a failure mid-generation
        # (e.g. MemoryError on large --qubits, or disk-full) still removes them.
        tmp_paths = [f64_path, f32_path]
        paths = {"float64": f64_path, "float32": f32_path}
    else:
        paths = {"float64": args.parquet, "float32": args.parquet}

    try:
        if args.parquet is None:
            print(
                f"Generating {total_rows} rows x {sample_size} cols (native f32 + f64)"
            )
            _generate_parquet(
                paths["float64"], total_rows, sample_size, np.float64, pa.float64()
            )
            _generate_parquet(
                paths["float32"], total_rows, sample_size, np.float32, pa.float32()
            )

        print("=" * 70)
        print("Parquet f32 vs f64 pipeline throughput")
        print(
            f"  qubits={args.qubits}, batches={args.batches}, "
            f"batch_size={args.batch_size}, encoding={args.encoding}"
        )
        print("=" * 70)

        results: dict[str, float] = {}
        for dtype in ("float64", "float32"):
            vps_trials: list[float] = []
            for t in range(args.trials):
                dur, vps = run_loader(
                    paths[dtype],
                    dtype,
                    args.qubits,
                    args.batches,
                    args.batch_size,
                    args.encoding,
                )
                vps_trials.append(vps)
                print(f"  [{dtype}] trial {t + 1}: {dur:.4f} s, {vps:.1f} vec/s")
            results[dtype] = sorted(vps_trials)[len(vps_trials) // 2]
            print(f"  [{dtype}] median: {results[dtype]:.1f} vec/s")

        print("-" * 70)
        if results["float64"] <= 0.0:
            print("  No vectors encoded (empty file?); skipping speedup.")
        else:
            speedup = results["float32"] / results["float64"]
            print(f"  f32/f64 speedup: {speedup:.3f}x ({(speedup - 1.0) * 100:+.1f}%)")
    finally:
        for p in tmp_paths:
            os.remove(p)


if __name__ == "__main__":
    main()
