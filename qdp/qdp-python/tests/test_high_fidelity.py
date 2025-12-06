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
Tests include: full-stack verification, async pipeline, fidelity metrics,
zero-copy validation, and edge cases (boundaries, stability, memory, threads).
"""

import pytest
import torch
import numpy as np
import concurrent.futures
from mahout_qdp import QdpEngine

np.random.seed(2026)

# ASYNC_THRESHOLD = 1MB / sizeof(f64) = 131072
PIPELINE_CHUNK_SIZE = 131072


def calculate_fidelity(
    state_vector_gpu: torch.Tensor, ground_truth_cpu: np.ndarray
) -> float:
    """Calculate quantum state fidelity: F = |<ψ_gpu | ψ_cpu>|²"""
    psi_gpu = state_vector_gpu.cpu().numpy()

    if np.any(np.isnan(psi_gpu)) or np.any(np.isinf(psi_gpu)):
        return 0.0

    assert psi_gpu.shape == ground_truth_cpu.shape, (
        f"Shape mismatch: {psi_gpu.shape} vs {ground_truth_cpu.shape}"
    )

    overlap = np.vdot(ground_truth_cpu, psi_gpu)
    fidelity = np.abs(overlap) ** 2
    return float(fidelity)


@pytest.fixture(scope="module")
def engine():
    """Initialize QDP engine (module-scoped singleton)."""
    try:
        return QdpEngine(0)
    except RuntimeError as e:
        pytest.skip(f"CUDA initialization failed: {e}")


# 1. Core Logic and Boundary Tests


@pytest.mark.gpu
@pytest.mark.parametrize(
    "num_qubits, data_size, desc",
    [
        (4, 16, "Small - Sync Path"),
        (10, 1000, "Medium - Padding Logic"),
        (18, PIPELINE_CHUNK_SIZE, "Boundary - Exact Chunk Size"),
        (18, PIPELINE_CHUNK_SIZE + 1, "Boundary - Chunk + 1"),
        (18, PIPELINE_CHUNK_SIZE * 2, "Boundary - Two Exact Chunks"),
        (20, 1_000_000, "Large - Async Pipeline"),
    ],
)
def test_amplitude_encoding_fidelity_comprehensive(engine, num_qubits, data_size, desc):
    """Test fidelity across sync path, async pipeline, and chunk boundaries."""
    print(f"\n[Test Case] {desc} (Size: {data_size})")

    raw_data = np.random.rand(data_size).astype(np.float64)
    norm = np.linalg.norm(raw_data)
    expected_state = raw_data / norm

    state_len = 1 << num_qubits
    if data_size < state_len:
        padding = np.zeros(state_len - data_size, dtype=np.float64)
        expected_state = np.concatenate([expected_state, padding])

    expected_state_complex = expected_state.astype(np.complex128)
    qtensor = engine.encode(raw_data.tolist(), num_qubits, "amplitude")
    torch_state = torch.from_dlpack(qtensor)

    assert torch_state.is_cuda, "Tensor must be on GPU"
    assert torch_state.dtype == torch.complex128, "Tensor must be Complex128"
    assert torch_state.shape[0] == state_len, "Tensor shape must match 2^n"

    fidelity = calculate_fidelity(torch_state, expected_state_complex)
    print(f"Fidelity: {fidelity:.16f}")

    assert fidelity > (1.0 - 1e-14), f"Fidelity loss in {desc}! F={fidelity}"


@pytest.mark.gpu
def test_complex_integrity(engine):
    """Verify imaginary part is strictly 0 for amplitude encoding."""
    num_qubits = 12
    data_size = 3000  # Non-power-of-2 size

    raw_data = np.random.rand(data_size).astype(np.float64)
    qtensor = engine.encode(raw_data.tolist(), num_qubits, "amplitude")
    torch_state = torch.from_dlpack(qtensor)

    imag_error = torch.sum(torch.abs(torch_state.imag)).item()
    print(f"\nSum of imaginary parts (should be 0): {imag_error}")
    assert imag_error == 0.0, "State vector contains non-zero imaginary components!"


# 2. Numerical Stability Tests


@pytest.mark.gpu
def test_numerical_stability_underflow(engine):
    """Test precision with extremely small values (1e-150)."""
    num_qubits = 4
    data = [1e-150] * 16

    qtensor = engine.encode(data, num_qubits, "amplitude")
    torch_state = torch.from_dlpack(qtensor)

    assert not torch.isnan(torch_state).any(), "Result contains NaN for small inputs"

    probs = torch.abs(torch_state) ** 2
    total_prob = torch.sum(probs).item()
    assert abs(total_prob - 1.0) < 1e-10, f"Normalization failed: {total_prob}"


# 3. Memory Leak Tests


@pytest.mark.gpu
def test_memory_leak_quantitative(engine):
    """Quantitative memory leak test using torch.cuda.memory_allocated()."""
    num_qubits = 10
    data = [0.1] * 1024
    iterations = 500

    _ = torch.from_dlpack(engine.encode(data, num_qubits, "amplitude"))
    torch.cuda.synchronize()

    start_mem = torch.cuda.memory_allocated()
    print(f"\nStart GPU Memory: {start_mem} bytes")

    for _ in range(iterations):
        qtensor = engine.encode(data, num_qubits, "amplitude")
        t = torch.from_dlpack(qtensor)
        del t
        del qtensor

    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated()
    print(f"End GPU Memory:   {end_mem} bytes")

    assert end_mem == start_mem, (
        f"Memory leak detected! Leaked {end_mem - start_mem} bytes"
    )


@pytest.mark.gpu
def test_memory_safety_stress(engine):
    """Stress test: rapid encode/release to verify DLPack deleter."""
    import gc

    num_qubits = 10
    data = [0.1] * 1024
    iterations = 1000

    print(f"\nStarting memory stress test ({iterations} iterations)...")

    for _ in range(iterations):
        qtensor = engine.encode(data, num_qubits, "amplitude")
        t = torch.from_dlpack(qtensor)
        del t
        del qtensor

    gc.collect()
    torch.cuda.empty_cache()
    print("Memory stress test passed (no crash).")


# 4. Thread Safety Tests


@pytest.mark.gpu
def test_multithreaded_access(engine):
    """Test concurrent access from multiple threads (validates Send+Sync)."""

    def worker_task(thread_id):
        size = 100 + thread_id
        data = np.random.rand(size).tolist()
        try:
            qtensor = engine.encode(data, 10, "amplitude")
            t = torch.from_dlpack(qtensor)
            return t.is_cuda
        except Exception as e:
            return e

    num_threads = 8
    print(f"\nStarting concurrent stress test with {num_threads} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, i) for i in range(num_threads)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if isinstance(result, Exception):
                pytest.fail(f"Thread failed with error: {result}")
            assert result is True, "Thread execution result invalid"

    print("Multithreaded access check passed.")


# 5. Error Propagation Tests


@pytest.mark.gpu
def test_error_propagation(engine):
    """Verify Rust errors are correctly propagated to Python RuntimeError."""
    with pytest.raises(RuntimeError, match="Input data cannot be empty|empty|Empty"):
        engine.encode([], 5, "amplitude")

    with pytest.raises(RuntimeError, match="at least 1|qubit|Qubit"):
        engine.encode([1.0], 0, "amplitude")

    with pytest.raises(RuntimeError, match="exceeds state vector size|exceed|capacity"):
        engine.encode([1.0, 1.0, 1.0], 1, "amplitude")
