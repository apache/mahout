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
DataLoader throughput benchmark across Mahout (QDP), PennyLane, and Qiskit.

The workload mirrors the `qdp-core/examples/dataloader_throughput.rs` pipeline:
- Generate batches of size `BATCH_SIZE` with deterministic vectors.
- Prefetch on the CPU side to keep the GPU fed.
- Encode vectors into amplitude states on GPU and run a tiny consumer op.

Frameworks:

* ``mahout``           — QDP Rust+CUDA via :class:`qumat_qdp.QdpBenchmark`.
* ``mahout-amd``       — QDP AMD path via :class:`qumat_qdp.QdpEngine` with
                         ``backend="amd"`` (TritonAmdEngine on ROCm).
* ``pennylane``        — PennyLane ``default.qubit`` (uses CUDA torch tensors
                         when available, otherwise CPU).
* ``pennylane-amdgpu`` — PennyLane ``lightning.amdgpu``, the official ROCm
                         simulator. Requires ``pennylane-lightning-amdgpu``
                         and a system ROCm 7.x install. Set the
                         ``ROCM_LIB_DIR`` env var BEFORE invoking python so
                         the matching HSA/HIP libs are preloaded — the
                         ctypes preload must happen before torch/pennylane
                         import to avoid a deadlock with the older
                         libhsa-runtime Ubuntu 24.04 ships at
                         /lib/x86_64-linux-gnu.
* ``pytorch-ref``      — Pure-PyTorch reference implementation
                         (:func:`qumat_qdp.torch_ref.amplitude_encode`).
                         Same workload, no engine wrapper — useful as a
                         "what naive PyTorch on the same hardware can do"
                         ceiling for the AMD comparison.
* ``qiskit``           — Qiskit Aer ``statevector`` simulator (CPU).

Run from qdp-python directory (qumat_qdp must be importable, e.g. via uv):
    uv run python benchmark/benchmark_throughput.py --qubits 16 --batches 200 --batch-size 64

    # AMD multi-framework comparison:
    ROCM_LIB_DIR=/opt/rocm-7.2.0/lib uv run python benchmark/benchmark_throughput.py \\
        --frameworks mahout-amd,pennylane,pennylane-amdgpu --qubits 12
"""

import argparse
import ctypes
import os
import time


# IMPORTANT: preload system ROCm 7.x libs *before* importing torch / pennylane.
# Once torch's HIP runtime maps the older libhsa-runtime from
# /lib/x86_64-linux-gnu (Ubuntu 24.04 ships ROCm 5.7), a later RTLD_GLOBAL of
# the newer libhsa deadlocks. Doing the preload at module top is the only
# reliable option short of relying on LD_LIBRARY_PATH being set externally.
def _preload_rocm_libs_at_import(lib_dir: str | None = None) -> None:
    candidate = lib_dir or os.environ.get("ROCM_LIB_DIR") or "/opt/rocm/lib"
    for name in ("libhsa-runtime64.so.1", "libamdhip64.so.7"):
        path = os.path.join(candidate, name)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


# Auto-preload only when the user explicitly opts in via env. Tightened to
# MAHOUT_PRELOAD_ROCM=1 only (NOT ROCM_LIB_DIR alone) so a stale exported
# ROCM_LIB_DIR doesn't auto-trigger global symbol injection in unrelated
# processes that import this module.
if os.environ.get("MAHOUT_PRELOAD_ROCM") == "1":
    _preload_rocm_libs_at_import()


import numpy as np
import torch
from qumat_qdp import QdpBenchmark

from benchmark.utils import normalize_batch, prefetched_batches

BAR = "=" * 70
SEP = "-" * 70
FRAMEWORK_CHOICES = (
    "pennylane",
    "qiskit",
    "mahout",
    "mahout-amd",
    "pennylane-amdgpu",
    "pytorch-ref",
)

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


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


WARMUP_BATCHES = 3


def run_mahout(
    num_qubits: int,
    total_batches: int,
    batch_size: int,
    prefetch: int,
    encoding_method: str = "amplitude",
):
    """Run Mahout throughput using the generic user API (QdpBenchmark)."""
    try:
        result = (
            QdpBenchmark(device_id=0)
            .qubits(num_qubits)
            .encoding(encoding_method)
            .batches(total_batches, size=batch_size)
            .prefetch(prefetch)
            .warmup(WARMUP_BATCHES)
            .run_throughput()
        )
    except Exception as exc:
        print(f"[Mahout] Init failed: {exc}")
        return 0.0, 0.0

    print(f"  IO + Encode Time: {result.duration_sec:.4f} s")
    print(
        f"  Total Time: {result.duration_sec:.4f} s "
        f"({result.vectors_per_sec:.1f} vectors/sec)"
    )
    return result.duration_sec, result.vectors_per_sec


def _sample_dim(num_qubits: int, encoding_method: str) -> int:
    if encoding_method == "basis":
        return 1
    if encoding_method in {"angle", "iqp-z"}:
        return num_qubits
    if encoding_method == "iqp":
        return num_qubits + num_qubits * (num_qubits - 1) // 2
    return 1 << num_qubits


def run_pennylane(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    if not HAS_PENNYLANE:
        print("[PennyLane] Not installed, skipping.")
        return 0.0, 0.0

    dev = qml.device("default.qubit", wires=num_qubits)
    cuda_available = torch.cuda.is_available()
    target_device = "cuda" if cuda_available else "cpu"

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    # Warmup: amortize QNode tracing, default.qubit JIT caches, and the first
    # CUDA stream kernel launch outside the timer so the headline number
    # reflects steady-state throughput, not first-batch initialization.
    for warmup_batch in prefetched_batches(
        WARMUP_BATCHES, batch_size, 1 << num_qubits, prefetch
    ):
        wb = torch.tensor(warmup_batch, dtype=torch.float32, device=target_device)
        _ = circuit(wb)
    if cuda_available:
        torch.cuda.synchronize()

    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        # float32 input keeps every framework on the same precision.  Returned
        # state is complex64 on the same device — no lossy cast back to float.
        batch_t = torch.tensor(batch, dtype=torch.float32, device=target_device)
        state = circuit(batch_t)
        _ = state.abs().sum()
        processed += len(batch)

    if cuda_available:
        torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_mahout_amd(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    """Run Mahout AMD path (TritonAmdEngine) end-to-end through the prefetch pipeline."""
    try:
        from qumat_qdp import QdpEngine, is_triton_amd_available
    except ImportError as exc:
        print(f"[Mahout-AMD] qumat_qdp unavailable: {exc}")
        return 0.0, 0.0

    if not is_triton_amd_available():
        print("[Mahout-AMD] Triton AMD backend unavailable on this host, skipping.")
        return 0.0, 0.0

    try:
        engine = QdpEngine(device_id=0, precision="float32", backend="amd")
    except Exception as exc:
        print(f"[Mahout-AMD] Init failed: {exc}")
        return 0.0, 0.0

    # Warmup: first encode triggers Triton AMD JIT autotune (~100s of ms).
    for warmup_batch in prefetched_batches(
        WARMUP_BATCHES, batch_size, 1 << num_qubits, prefetch
    ):
        wb = torch.tensor(warmup_batch, dtype=torch.float32, device="cuda")
        _ = engine.encode(wb, num_qubits, "amplitude")
    torch.cuda.synchronize()

    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        batch_t = torch.tensor(batch, dtype=torch.float32, device="cuda")
        state = engine.encode(batch_t, num_qubits, "amplitude")
        _ = state.abs().sum()
        processed += len(batch)

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_pytorch_ref(
    num_qubits: int, total_batches: int, batch_size: int, prefetch: int
):
    """Run the project's PyTorch-only reference implementation.

    Same amplitude-encoding workload as `mahout-amd`, but goes through
    `qumat_qdp.torch_ref.amplitude_encode` (a pure PyTorch op chain:
    L2 normalize, zero-pad to 2**num_qubits, complex view) with no
    engine wrapper. Useful as a ceiling for "what naive PyTorch on the
    same hardware can do" — gaps between this and `mahout-amd` quantify
    the per-call overhead in the AMD engine adapter.
    """
    try:
        from qumat_qdp.torch_ref import amplitude_encode
    except ImportError as exc:
        print(f"[PyTorch-ref] qumat_qdp.torch_ref unavailable: {exc}")
        return 0.0, 0.0

    cuda_available = torch.cuda.is_available()
    target_device = "cuda" if cuda_available else "cpu"

    # Warmup: amortize first kernel launches.
    for warmup_batch in prefetched_batches(
        WARMUP_BATCHES, batch_size, 1 << num_qubits, prefetch
    ):
        wb = torch.tensor(warmup_batch, dtype=torch.float32, device=target_device)
        _ = amplitude_encode(wb, num_qubits)
    if cuda_available:
        torch.cuda.synchronize()

    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        batch_t = torch.tensor(batch, dtype=torch.float32, device=target_device)
        state = amplitude_encode(batch_t, num_qubits)
        _ = state.abs().sum()
        processed += len(batch)

    if cuda_available:
        torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_pennylane_amdgpu(
    num_qubits: int, total_batches: int, batch_size: int, prefetch: int
):
    """Run PennyLane lightning.amdgpu (native ROCm Kokkos+HIP simulator).

    Note: lightning.amdgpu is the official PennyLane ROCm simulator but does
    not broadcast over the batch dimension for AmplitudeEmbedding, so this
    runner uses a per-sample loop. The throughput reflects what a user gets
    from the public API today; the gap to Mahout-AMD is dominated by
    PennyLane QNode dispatch overhead per sample, not by the underlying
    Kokkos+HIP kernel.
    """
    if not HAS_PENNYLANE:
        print("[PennyLane-AMDGPU] PennyLane not installed, skipping.")
        return 0.0, 0.0

    try:
        dev = qml.device("lightning.amdgpu", wires=num_qubits)
    except Exception as exc:
        print(f"[PennyLane-AMDGPU] Device init failed: {exc}")
        print(
            "  Hint: launch with MAHOUT_PRELOAD_ROCM=1 ROCM_LIB_DIR=/opt/rocm-X/lib "
            "(or the dir containing libhsa-runtime64.so.1 + libamdhip64.so.7) so "
            "the matching ROCm libs are RTLD_GLOBAL-preloaded at module top."
        )
        return 0.0, 0.0

    @qml.qnode(dev)
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    # Warmup: amortize Kokkos device init + first-call QNode tracing.
    for warmup_batch in prefetched_batches(
        WARMUP_BATCHES, batch_size, 1 << num_qubits, prefetch
    ):
        for vec in warmup_batch:
            _ = circuit(np.asarray(vec, dtype=np.float32))
    torch.cuda.synchronize()

    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        states = []
        for vec in batch:
            # float32 input matches Mahout-AMD precision.  lightning.amdgpu
            # casts internally to whatever the simulator was configured for.
            states.append(circuit(np.asarray(vec, dtype=np.float32)))
            processed += 1
        gpu_tensor = torch.tensor(
            np.array(states), device="cuda", dtype=torch.complex64
        )
        _ = gpu_tensor.abs().sum()

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_qiskit(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    if not HAS_QISKIT:
        print("[Qiskit] Not installed, skipping.")
        return 0.0, 0.0

    backend = AerSimulator(method="statevector")
    cuda_available = torch.cuda.is_available()

    # Warmup: amortize Qiskit transpile-cache + AerSimulator first-call cost.
    for warmup_batch in prefetched_batches(
        WARMUP_BATCHES, batch_size, 1 << num_qubits, prefetch
    ):
        normalized = normalize_batch(warmup_batch)
        for vec in normalized:
            qc = QuantumCircuit(num_qubits)
            qc.initialize(vec, range(num_qubits))
            qc.save_statevector()
            t_qc = transpile(qc, backend)
            backend.run(t_qc).result()
    if cuda_available:
        torch.cuda.synchronize()

    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        normalized = normalize_batch(batch)

        batch_states = []
        for vec in normalized:
            qc = QuantumCircuit(num_qubits)
            qc.initialize(vec, range(num_qubits))
            qc.save_statevector()
            t_qc = transpile(qc, backend)
            state = backend.run(t_qc).result().get_statevector().data
            batch_states.append(state)
            processed += 1

        if cuda_available:
            gpu_tensor = torch.tensor(
                np.array(batch_states), device="cuda", dtype=torch.complex64
            )
            _ = gpu_tensor.abs().sum()
        else:
            cpu_tensor = torch.tensor(np.array(batch_states), dtype=torch.complex64)
            _ = cpu_tensor.abs().sum()

    if cuda_available:
        torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"\n  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DataLoader throughput across frameworks."
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=16,
        help="Number of qubits (power-of-two vector length).",
    )
    parser.add_argument(
        "--batches", type=int, default=200, help="Total batches to stream."
    )
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
            "(pennylane, pennylane-amdgpu, pytorch-ref, qiskit, mahout, mahout-amd) "
            "or 'all'."
        ),
    )
    parser.add_argument(
        "--encoding-method",
        type=str,
        default="amplitude",
        choices=["amplitude", "angle", "basis", "iqp", "iqp-z"],
        help="Encoding method to use for Mahout (amplitude, angle, basis, iqp, or iqp-z).",
    )
    parser.add_argument(
        "--rocm-lib-dir",
        type=str,
        default=None,
        help=(
            "DEPRECATED: set ROCM_LIB_DIR env var BEFORE invoking the script "
            "(or LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib python -m ...). The "
            "ctypes preload must happen before torch/pennylane import to be "
            "effective; passing the dir on the CLI runs after imports and "
            "deadlocks. This flag prints a hint and otherwise does nothing."
        ),
    )
    args = parser.parse_args()

    if args.rocm_lib_dir and os.environ.get("ROCM_LIB_DIR") != args.rocm_lib_dir:
        print(
            f"NOTE: --rocm-lib-dir={args.rocm_lib_dir} ignored at this stage; "
            "set it via ROCM_LIB_DIR env var before launching python.\n"
            f"  Try: ROCM_LIB_DIR={args.rocm_lib_dir} python -m benchmark.benchmark_throughput ..."
        )

    try:
        frameworks = parse_frameworks(args.frameworks)
    except ValueError as exc:
        parser.error(str(exc))

    # TODO: fix this with #1252 in the future.
    if args.encoding_method in {"iqp", "iqp-z"}:
        unsupported = [name for name in frameworks if name != "mahout"]
        if unsupported:
            print(
                "Warning: IQP benchmarks in this script currently support only "
                "framework 'mahout'; skipping unsupported frameworks: "
                f"{', '.join(unsupported)}."
            )
            frameworks = ["mahout"]

    total_vectors = args.batches * args.batch_size
    vector_len = _sample_dim(args.qubits, args.encoding_method)

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
        f"DATALOADER THROUGHPUT BENCHMARK: {args.qubits} Qubits, {total_vectors} Samples"
    )
    print(BAR)

    t_pl = th_pl = t_qiskit = th_qiskit = t_mahout = th_mahout = 0.0
    t_pl_amd = th_pl_amd = t_mahout_amd = th_mahout_amd = 0.0
    t_pt_ref = th_pt_ref = 0.0

    if "pennylane" in frameworks:
        print()
        print("[PennyLane] Full Pipeline (DataLoader -> GPU)...")
        t_pl, th_pl = run_pennylane(
            args.qubits, args.batches, args.batch_size, args.prefetch
        )

    if "pennylane-amdgpu" in frameworks:
        print()
        print("[PennyLane-AMDGPU] Full Pipeline (lightning.amdgpu, per-sample)...")
        t_pl_amd, th_pl_amd = run_pennylane_amdgpu(
            args.qubits, args.batches, args.batch_size, args.prefetch
        )

    if "pytorch-ref" in frameworks:
        print()
        print("[PyTorch-ref] Full Pipeline (qumat_qdp.torch_ref.amplitude_encode)...")
        t_pt_ref, th_pt_ref = run_pytorch_ref(
            args.qubits, args.batches, args.batch_size, args.prefetch
        )

    if "qiskit" in frameworks:
        print()
        print("[Qiskit] Full Pipeline (DataLoader -> GPU)...")
        t_qiskit, th_qiskit = run_qiskit(
            args.qubits, args.batches, args.batch_size, args.prefetch
        )

    if "mahout" in frameworks:
        print()
        print("[Mahout] Full Pipeline (DataLoader -> GPU)...")
        t_mahout, th_mahout = run_mahout(
            args.qubits,
            args.batches,
            args.batch_size,
            args.prefetch,
            args.encoding_method,
        )

    if "mahout-amd" in frameworks:
        print()
        print("[Mahout-AMD] Full Pipeline (TritonAmdEngine, ROCm)...")
        t_mahout_amd, th_mahout_amd = run_mahout_amd(
            args.qubits, args.batches, args.batch_size, args.prefetch
        )

    print()
    print(BAR)
    print("THROUGHPUT (Higher is Better)")
    print(f"Samples: {total_vectors}, Qubits: {args.qubits}")
    print(BAR)

    throughput_results = []
    if th_pl > 0:
        throughput_results.append(("PennyLane", th_pl))
    if th_pl_amd > 0:
        throughput_results.append(("PennyLane-AMDGPU", th_pl_amd))
    if th_pt_ref > 0:
        throughput_results.append(("PyTorch-ref", th_pt_ref))
    if th_qiskit > 0:
        throughput_results.append(("Qiskit", th_qiskit))
    if th_mahout > 0:
        throughput_results.append(("Mahout", th_mahout))
    if th_mahout_amd > 0:
        throughput_results.append(("Mahout-AMD", th_mahout_amd))

    throughput_results.sort(key=lambda x: x[1], reverse=True)

    for name, tput in throughput_results:
        print(f"{name:18s} {tput:10.1f} vectors/sec")

    if t_mahout > 0 or t_mahout_amd > 0:
        print(SEP)
        # Prefer the available Mahout reference for ratio reporting.
        ref_name, ref_th = (
            ("Mahout", th_mahout) if t_mahout > 0 else ("Mahout-AMD", th_mahout_amd)
        )
        if t_pl > 0:
            print(f"Speedup {ref_name} vs PennyLane:        {ref_th / th_pl:10.2f}x")
        if t_pl_amd > 0:
            print(
                f"Speedup {ref_name} vs PennyLane-AMDGPU: {ref_th / th_pl_amd:10.2f}x"
            )
        if t_pt_ref > 0:
            print(
                f"Speedup {ref_name} vs PyTorch-ref:      {ref_th / th_pt_ref:10.2f}x"
            )
        if t_qiskit > 0:
            print(
                f"Speedup {ref_name} vs Qiskit:           {ref_th / th_qiskit:10.2f}x"
            )
        if t_mahout > 0 and t_mahout_amd > 0:
            print(
                f"Mahout (CUDA) vs Mahout-AMD:           {th_mahout / th_mahout_amd:10.2f}x"
            )


if __name__ == "__main__":
    main()
