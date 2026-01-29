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

"""
Baseline benchmark driver for QDP optimization.

Runs throughput and latency benchmarks multiple times (default 5), computes
median/p95, gathers system metadata, and writes CSV + markdown report to
qdp/docs/optimization/results/.

Set observability before running (recommended):
  export QDP_ENABLE_POOL_METRICS=1
  export QDP_ENABLE_OVERLAP_TRACKING=1
  export RUST_LOG=info

Usage:
  cd qdp/qdp-python/benchmark
  uv run python run_pipeline_baseline.py --qubits 16 --batch-size 64 --prefetch 16 --batches 500 --trials 20
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Set observability env before importing Rust-backed modules (so pipeline sees them)
os.environ.setdefault("QDP_ENABLE_POOL_METRICS", "1")
os.environ.setdefault("QDP_ENABLE_OVERLAP_TRACKING", "1")
os.environ.setdefault("RUST_LOG", "info")

from benchmark_latency import run_mahout as run_mahout_latency
from benchmark_throughput import run_mahout as run_mahout_throughput


def _repo_root() -> Path:
    # benchmark -> qdp-python -> qdp -> mahout (workspace root)
    return Path(__file__).resolve().parent.parent.parent.parent


def _results_dir() -> Path:
    return _repo_root() / "qdp" / "docs" / "optimization" / "results"


def get_git_commit(repo_root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def get_gpu_info() -> tuple[str, str, str]:
    gpu = driver = cuda = "unknown"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0) or "unknown"
            cuda = getattr(torch.version, "cuda", None) or "unknown"
        # Driver from nvidia-smi if available
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout:
            line = r.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 1 and not gpu or gpu == "unknown":
                gpu = parts[0]
            if len(parts) >= 2:
                driver = parts[1]
    except Exception:
        pass
    return gpu, driver, cuda


def run_throughput_trials(
    qubits: int,
    batches: int,
    batch_size: int,
    prefetch: int,
    trials: int,
    encoding: str,
) -> list[float]:
    throughputs: list[float] = []
    for i in range(trials):
        _duration, throughput = run_mahout_throughput(
            qubits, batches, batch_size, prefetch, encoding
        )
        if throughput > 0:
            throughputs.append(throughput)
    return throughputs


def run_latency_trials(
    qubits: int,
    batches: int,
    batch_size: int,
    prefetch: int,
    trials: int,
    encoding: str,
) -> list[float]:
    latencies_ms: list[float] = []
    for i in range(trials):
        _duration, latency_ms = run_mahout_latency(
            qubits, batches, batch_size, prefetch, encoding
        )
        if latency_ms > 0:
            latencies_ms.append(latency_ms)
    return latencies_ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline benchmarks and write CSV + report."
    )
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--encoding-method",
        type=str,
        default="amplitude",
        choices=["amplitude", "angle", "basis"],
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="pipeline_baseline",
        help="Prefix for output files (e.g. pipeline_baseline -> pipeline_baseline_YYYYMMDD_rep_config).",
    )
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Skip throughput trials.",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency trials.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    results_dir = _results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.utcnow().strftime("%Y%m%d")
    config_tag = "rep_config"
    base_name = f"{args.output_prefix}_{date_str}_{config_tag}"

    commit = get_git_commit(repo_root)
    gpu, driver, cuda = get_gpu_info()

    throughputs: list[float] = []
    latencies_ms: list[float] = []

    if not args.skip_throughput:
        print(
            f"Running throughput: {args.trials} trials (qubits={args.qubits}, batch_size={args.batch_size}, prefetch={args.prefetch}, batches={args.batches})"
        )
        throughputs = run_throughput_trials(
            args.qubits,
            args.batches,
            args.batch_size,
            args.prefetch,
            args.trials,
            args.encoding_method,
        )
        if throughputs:
            print(
                f"  Throughput: median={np.median(throughputs):.1f} vec/s, p95={np.percentile(throughputs, 95):.1f} vec/s"
            )

    if not args.skip_latency:
        print(f"Running latency: {args.trials} trials")
        latencies_ms = run_latency_trials(
            args.qubits,
            args.batches,
            args.batch_size,
            args.prefetch,
            args.trials,
            args.encoding_method,
        )
        if latencies_ms:
            print(
                f"  Latency: median={np.median(latencies_ms):.3f} ms/vec, p95={np.percentile(latencies_ms, 95):.3f} ms/vec"
            )

    # Stats (used in markdown report)
    throughput_median = float(np.median(throughputs)) if throughputs else 0.0
    throughput_p95 = float(np.percentile(throughputs, 95)) if throughputs else 0.0
    latency_p50 = float(np.median(latencies_ms)) if latencies_ms else 0.0
    latency_p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    date_iso = datetime.utcnow().strftime("%Y-%m-%d")

    # Markdown report
    md_lines = [
        "# pipeline baseline report",
        "",
        f"- **Date**: {date_iso}",
        f"- **Git commit**: {commit}",
        f"- **GPU**: {gpu}",
        f"- **Driver**: {driver}",
        f"- **CUDA**: {cuda}",
        "",
        "## Parameters",
        "",
        f"- qubits: {args.qubits}",
        f"- batch_size: {args.batch_size}",
        f"- prefetch: {args.prefetch}",
        f"- batches: {args.batches}",
        f"- trials: {args.trials}",
        f"- encoding: {args.encoding_method}",
        "",
        "## Results",
        "",
        "| Metric | Median | P95 |",
        "|--------|--------|-----|",
        f"| Throughput (vectors/sec) | {throughput_median:.1f} | {throughput_p95:.1f} |",
        f"| Latency (ms/vector) | {latency_p50:.3f} | {latency_p95:.3f} |",
        "",
        "---",
        "",
        "*Generated by run_pipeline_baseline.py*",
    ]
    md_path = results_dir / f"{base_name}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
