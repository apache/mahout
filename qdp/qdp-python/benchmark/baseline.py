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

"""Capture and compare a performance baseline for every encoding.

Numbers are machine-specific, so the JSON is meant to live outside the repo
(a CI artifact or a scratch directory), not be committed.

    # Before an engine refactor:
    python baseline.py capture --out /tmp/qdp-baseline.json

    # After:
    python baseline.py compare /tmp/qdp-baseline.json --tolerance 0.05

``compare`` exits non-zero when any encoding's throughput drops by more than
``--tolerance`` (a fraction) or its latency rises by more than the same
fraction. Each cell is measured ``--repeats`` times and the best run is kept,
which removes most of the noise from a shared machine.

An ``f32`` cell is only recorded for encodings with a native float32 pipeline
path. The pipeline silently measures float64 for the others, so recording
those under an ``f32`` key would make a future real f32 implementation compare
against a float64 baseline. Skipped cells are listed in the JSON under
``"skipped"`` and are ignored by ``compare``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from typing import Any

from qumat_qdp import QdpBenchmark
from qumat_qdp._backend import get_qdp

ENCODINGS = ("amplitude", "angle", "basis", "iqp", "iqp-z", "phase")
DTYPES = ("f64", "f32")


def supported_cells() -> tuple[list[str], list[str]]:
    """Split the encoding x dtype grid into measurable and skipped cell keys.

    A cell is skipped when the native pipeline has no float32 path for the
    encoding and would silently measure float64 instead.
    """
    qdp = get_qdp()
    if qdp is None:
        raise RuntimeError("The _qdp extension is required to capture a baseline")
    cells: list[str] = []
    skipped: list[str] = []
    for encoding in ENCODINGS:
        for dtype in DTYPES:
            key = f"{encoding}/{dtype}"
            if dtype == "f32" and not qdp.encoding_supports_f32(encoding):
                skipped.append(key)
            else:
                cells.append(key)
    return cells, skipped


def _gpu_name(device_id: int) -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # informational only
        return "unknown"


def measure(args: argparse.Namespace) -> dict[str, Any]:
    keys, skipped = supported_cells()
    cells: dict[str, dict[str, float]] = {}
    for key in keys:
        encoding, dtype = key.split("/")
        bench = (
            QdpBenchmark(device_id=args.device)
            .qubits(args.qubits)
            .encoding(encoding)
            .batches(args.batches, args.batch_size)
            .warmup(args.warmup)
            .dtype(dtype)
        )
        best_tp = 0.0
        best_lat = float("inf")
        for _ in range(args.repeats):
            tp = bench.run_throughput()
            lat = bench.run_latency()
            best_tp = max(best_tp, tp.vectors_per_sec)
            best_lat = min(best_lat, lat.latency_ms_per_vector)
        cells[key] = {"vectors_per_sec": best_tp, "latency_ms_per_vector": best_lat}
        print(
            f"{key:<16} {best_tp:>14,.0f} vec/s   {best_lat:>10.5f} ms/vec",
            flush=True,
        )
    for key in skipped:
        print(f"{key:<16} {'skipped':>14}   no native f32 path (would measure f64)")
    return {
        "config": {
            "qubits": args.qubits,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "host": platform.node(),
        "gpu": _gpu_name(args.device),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cells": cells,
        "skipped": skipped,
    }


def cmd_capture(args: argparse.Namespace) -> int:
    result = measure(args)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nWrote {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    with open(args.baseline, encoding="utf-8") as fh:
        baseline = json.load(fh)
    cfg = baseline["config"]
    for name in ("qubits", "batches", "batch_size", "warmup", "repeats"):
        setattr(args, name, cfg[name])
    current = measure(args)

    print()
    print(f"{'cell':<16} {'throughput Δ':>14} {'latency Δ':>12}   status")
    failed = False
    for key, base in baseline["cells"].items():
        now = current["cells"].get(key)
        if now is None:
            print(f"{key:<16} {'missing':>14}")
            failed = True
            continue
        tp_delta = now["vectors_per_sec"] / base["vectors_per_sec"] - 1.0
        lat_delta = now["latency_ms_per_vector"] / base["latency_ms_per_vector"] - 1.0
        bad = tp_delta < -args.tolerance or lat_delta > args.tolerance
        failed |= bad
        print(
            f"{key:<16} {tp_delta:>+13.1%} {lat_delta:>+11.1%}   {'REGRESSION' if bad else 'ok'}"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=2)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser(
        "capture", help="measure every encoding and write a baseline JSON"
    )
    cap.add_argument("--out", required=True)
    cap.add_argument("--device", type=int, default=0)
    cap.add_argument("--qubits", type=int, default=16)
    cap.add_argument("--batches", type=int, default=40)
    cap.add_argument("--batch-size", type=int, default=64)
    cap.add_argument("--warmup", type=int, default=5)
    cap.add_argument("--repeats", type=int, default=3)
    cap.set_defaults(func=cmd_capture)

    cmp_ = sub.add_parser(
        "compare", help="re-measure with the baseline's config and report deltas"
    )
    cmp_.add_argument("baseline")
    cmp_.add_argument("--device", type=int, default=0)
    cmp_.add_argument("--tolerance", type=float, default=0.05)
    cmp_.add_argument("--out", help="also write the new measurement to this path")
    cmp_.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
