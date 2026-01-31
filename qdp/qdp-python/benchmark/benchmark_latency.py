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
Data-to-State latency benchmark: CPU RAM -> GPU VRAM.

Run:
    python qdp/qdp-python/benchmark/benchmark_latency.py --qubits 16 \
        --batches 200 --batch-size 64 --prefetch 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root to path so qumat_qdp is importable when run as script
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmark.utils import normalize_batch, prefetched_batches  # noqa: E402
from qumat_qdp import QdpBenchmark  # noqa: E402

BAR = "=" * 70
SEP = "-" * 70
FRAMEWORK_CHOICES = ("pennylane", "qiskit-init", "qiskit-statevector", "mahout")
FRAMEWORK_LABELS = {
    "mahout": "Mahout",
    "pennylane": "PennyLane",
    "qiskit-init": "Qiskit Initialize",
    "qiskit-statevector": "Qiskit Statevector",
}

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_frameworks(raw: str) -> list[str]:
    if raw.lower() == "all":
        return list(FRAMEWORK_CHOICES)

    selected: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in FRAMEWORK_CHOICES:
            raise ValueError(
                f"Unknown framework '{name}'. Choose from: "
                f"{', '.join(FRAMEWORK_CHOICES)} or 'all'."
            )
        if name not in selected:
            selected.append(name)

    return selected if selected else list(FRAMEWORK_CHOICES)


def run_mahout(
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    prefetch: int,
    encoding_method: str = "amplitude",
):
    """Run Mahout latency using the generic user API (QdpBenchmark)."""
    try:
        result = (
            QdpBenchmark(device_id=0)
            .qubits(num_qubits)
            .encoding(encoding_method)
            .batches(total_batches, size=batch_size)
            .prefetch(prefetch)
            .run_latency()
        )
    except Exception as exc:
        print(f"[Mahout] Init failed: {exc}")
        return 0.0, 0.0

    print(
        f"  Total Time: {result.duration_sec:.4f} s "
        f"({result.latency_ms_per_vector:.3f} ms/vector)"
    )
    return result.duration_sec, result.latency_ms_per_vector


def run_pennylane(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    if not HAS_PENNYLANE:
        print("[PennyLane] Not installed, skipping.")
        return 0.0, 0.0

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    sync_cuda()
    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        batch_cpu = torch.tensor(batch, dtype=torch.float64)
        try:
            state_cpu = circuit(batch_cpu)
        except Exception:
            state_cpu = torch.stack([circuit(x) for x in batch_cpu])
        _ = state_cpu.to("cuda", dtype=torch.complex64)
        processed += len(batch_cpu)

    sync_cuda()
    duration = time.perf_counter() - start
    latency_ms = (duration / processed) * 1000 if processed > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({latency_ms:.3f} ms/vector)")
    return duration, latency_ms


def run_qiskit_init(
    num_qubits: int, total_batches: int, batch_size: int, prefetch: int
):
    if not HAS_QISKIT:
        print("[Qiskit] Not installed, skipping.")
        return 0.0, 0.0

    backend = AerSimulator(method="statevector")
    sync_cuda()
    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        normalized = normalize_batch(batch)
        for vec in normalized:
            qc = QuantumCircuit(num_qubits)
            qc.initialize(vec, range(num_qubits))
            qc.save_statevector()
            t_qc = transpile(qc, backend)
            state = backend.run(t_qc).result().get_statevector().data
            _ = torch.tensor(state, device="cuda", dtype=torch.complex64)
            processed += 1

    sync_cuda()
    duration = time.perf_counter() - start
    latency_ms = (duration / processed) * 1000 if processed > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({latency_ms:.3f} ms/vector)")
    return duration, latency_ms


def run_qiskit_statevector(
    num_qubits: int, total_batches: int, batch_size: int, prefetch: int
):
    if not HAS_QISKIT:
        print("[Qiskit] Not installed, skipping.")
        return 0.0, 0.0

    sync_cuda()
    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        normalized = normalize_batch(batch)
        for vec in normalized:
            state = Statevector(vec)
            _ = torch.tensor(state.data, device="cuda", dtype=torch.complex64)
            processed += 1

    sync_cuda()
    duration = time.perf_counter() - start
    latency_ms = (duration / processed) * 1000 if processed > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({latency_ms:.3f} ms/vector)")
    return duration, latency_ms


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Data-to-State latency across frameworks."
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=16,
        help="Number of qubits (power-of-two vector length).",
    )
    parser.add_argument("--batches", type=int, default=200, help="Total batches.")
    parser.add_argument("--batch-size", type=int, default=64, help="Vectors per batch.")
    parser.add_argument(
        "--prefetch", type=int, default=16, help="CPU-side prefetch depth."
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default="all",
        help=(
            "Comma-separated list of frameworks to run "
            "(pennylane,qiskit-init,qiskit-statevector,mahout) or 'all'."
        ),
    )
    parser.add_argument(
        "--encoding-method",
        type=str,
        default="amplitude",
        choices=["amplitude", "angle", "basis"],
        help="Encoding method to use for Mahout (amplitude, angle, or basis).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device not available; GPU is required.")

    try:
        frameworks = parse_frameworks(args.frameworks)
    except ValueError as exc:
        parser.error(str(exc))

    total_vectors = args.batches * args.batch_size
    vector_len = 1 << args.qubits

    print(f"Generating {total_vectors} samples of {args.qubits} qubits...")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Vector length: {vector_len}")
    print(f"  Batches      : {args.batches}")
    print(f"  Prefetch     : {args.prefetch}")
    print(f"  Frameworks   : {', '.join(frameworks)}")
    print(f"  Encode method: {args.encoding_method}")
    bytes_per_vec = vector_len * 8
    print(f"  Generated {total_vectors} samples")
    print(
        f"  PennyLane/Qiskit format: {total_vectors * bytes_per_vec / (1024 * 1024):.2f} MB"
    )
    print(f"  Mahout format: {total_vectors * bytes_per_vec / (1024 * 1024):.2f} MB")
    print()

    print(BAR)
    print(
        f"DATA-TO-STATE LATENCY BENCHMARK: {args.qubits} Qubits, {total_vectors} Samples"
    )
    print(BAR)

    t_pl = l_pl = 0.0
    t_q_init = l_q_init = 0.0
    t_q_sv = l_q_sv = 0.0
    t_mahout = l_mahout = 0.0

    if "pennylane" in frameworks:
        print()
        print("[PennyLane] Full Pipeline (DataLoader -> GPU)...")
        t_pl, l_pl = run_pennylane(
            args.qubits, args.batches, args.batch_size, args.prefetch
        )

    if "qiskit-init" in frameworks:
        print()
        print("[Qiskit Initialize] Full Pipeline (DataLoader -> GPU)...")
        t_q_init, l_q_init = run_qiskit_init(
            args.qubits, args.batches, args.batch_size, args.prefetch
        )

    if "qiskit-statevector" in frameworks:
        print()
        print("[Qiskit Statevector] Full Pipeline (DataLoader -> GPU)...")
        t_q_sv, l_q_sv = run_qiskit_statevector(
            args.qubits, args.batches, args.batch_size, args.prefetch
        )

    if "mahout" in frameworks:
        print()
        print("[Mahout] Full Pipeline (DataLoader -> GPU)...")
        t_mahout, l_mahout = run_mahout(
            args.qubits,
            args.batches,
            args.batch_size,
            args.prefetch,
            args.encoding_method,
        )

    print()
    print(BAR)
    print("LATENCY (Lower is Better)")
    print(f"Samples: {total_vectors}, Qubits: {args.qubits}")
    print(BAR)

    latency_results = []
    if l_pl > 0:
        latency_results.append((FRAMEWORK_LABELS["pennylane"], l_pl))
    if l_q_init > 0:
        latency_results.append((FRAMEWORK_LABELS["qiskit-init"], l_q_init))
    if l_q_sv > 0:
        latency_results.append((FRAMEWORK_LABELS["qiskit-statevector"], l_q_sv))
    if l_mahout > 0:
        latency_results.append((FRAMEWORK_LABELS["mahout"], l_mahout))

    latency_results.sort(key=lambda x: x[1])

    for name, latency in latency_results:
        print(f"{name:18s} {latency:10.3f} ms/vector")

    if l_mahout > 0:
        print(SEP)
        if l_pl > 0:
            print(f"Speedup vs PennyLane:      {l_pl / l_mahout:10.2f}x")
        if l_q_init > 0:
            print(f"Speedup vs Qiskit Init:     {l_q_init / l_mahout:10.2f}x")
        if l_q_sv > 0:
            print(f"Speedup vs Qiskit Statevec: {l_q_sv / l_mahout:10.2f}x")


if __name__ == "__main__":
    main()
