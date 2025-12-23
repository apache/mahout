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
NumPy format I/O + Encoding benchmark: Mahout vs PennyLane

Compares the performance of loading quantum state data from NumPy .npy files
and encoding them on GPU between Mahout QDP and PennyLane.

Workflow:
1. Generate NumPy arrays with quantum state vectors
2. Save to .npy file
3. Load from file and encode on GPU
4. Measure total throughput (I/O + encoding)

Run:
    python qdp/benchmark/benchmark_numpy_io.py --qubits 10 --samples 1000
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from mahout_qdp import QdpEngine

BAR = "=" * 70
SEP = "-" * 70

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def generate_test_data(num_samples: int, sample_size: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic test data."""
    rng = np.random.RandomState(seed)
    data = rng.randn(num_samples, sample_size).astype(np.float64)
    # Normalize each sample
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return data / norms


def run_mahout_numpy(num_qubits: int, num_samples: int, npy_path: str):
    """Benchmark Mahout with NumPy file I/O."""
    print("\n[Mahout + NumPy] Loading and encoding...")
    
    try:
        engine = QdpEngine(0)
    except Exception as exc:
        print(f"  Init failed: {exc}")
        return 0.0, 0.0, 0.0
    
    # Measure file I/O + encoding together
    torch.cuda.synchronize()
    start_total = time.perf_counter()
    
    try:
        # Use the NumPy reader API
        qtensor = engine.encode_from_numpy(npy_path, num_qubits, "amplitude")
        tensor = torch.utils.dlpack.from_dlpack(qtensor)
        
        # Small computation to ensure GPU has processed the data
        _ = tensor.abs().sum()
        
        torch.cuda.synchronize()
        duration_total = time.perf_counter() - start_total
        
        throughput = num_samples / duration_total if duration_total > 0 else 0.0
        
        print(f"  Total Time (I/O + Encode): {duration_total:.4f} s")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Average per sample: {duration_total / num_samples * 1000:.2f} ms")
        
        return duration_total, throughput, duration_total / num_samples
        
    except Exception as exc:
        print(f"  Error: {exc}")
        return 0.0, 0.0, 0.0


def run_pennylane_numpy(num_qubits: int, num_samples: int, npy_path: str):
    """Benchmark PennyLane with NumPy file I/O."""
    if not HAS_PENNYLANE:
        print("\n[PennyLane + NumPy] Not installed, skipping.")
        return 0.0, 0.0, 0.0
    
    print("\n[PennyLane + NumPy] Loading and encoding...")
    
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()
    
    torch.cuda.synchronize()
    start_total = time.perf_counter()
    
    try:
        # Load NumPy file
        data = np.load(npy_path)
        
        # Process each sample
        states = []
        for i in range(len(data)):
            sample = torch.tensor(data[i], dtype=torch.float64)
            state = circuit(sample)
            states.append(state)
        
        # Move to GPU
        states_gpu = torch.stack(states).to("cuda", dtype=torch.complex64)
        _ = states_gpu.abs().sum()
        
        torch.cuda.synchronize()
        duration_total = time.perf_counter() - start_total
        
        throughput = num_samples / duration_total if duration_total > 0 else 0.0
        
        print(f"  Total Time (I/O + Encode): {duration_total:.4f} s")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Average per sample: {duration_total / num_samples * 1000:.2f} ms")
        
        return duration_total, throughput, duration_total / num_samples
        
    except Exception as exc:
        print(f"  Error: {exc}")
        return 0.0, 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NumPy I/O + Encoding: Mahout vs PennyLane"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=10,
        help="Number of qubits (vector length = 2^qubits)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save .npy file (default: temp file)",
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default="all",
        help="Comma-separated list: mahout,pennylane or 'all'",
    )
    args = parser.parse_args()
    
    # Parse frameworks
    if args.frameworks.lower() == "all":
        frameworks = ["mahout", "pennylane"]
    else:
        frameworks = [f.strip().lower() for f in args.frameworks.split(",")]
    
    num_qubits = args.qubits
    num_samples = args.samples
    sample_size = 1 << num_qubits  # 2^qubits
    
    print(BAR)
    print(f"NUMPY I/O + ENCODING BENCHMARK")
    print(BAR)
    print(f"Qubits: {num_qubits}")
    print(f"Sample size: {sample_size} elements")
    print(f"Number of samples: {num_samples}")
    print(f"Total data: {num_samples * sample_size * 8 / (1024**2):.2f} MB")
    print(f"Frameworks: {', '.join(frameworks)}")
    
    # Generate test data
    print("\nGenerating test data...")
    data = generate_test_data(num_samples, sample_size)
    
    # Save to NumPy file
    if args.output:
        npy_path = args.output
    else:
        fd, npy_path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
    
    print(f"Saving to {npy_path}...")
    np.save(npy_path, data)
    file_size_mb = os.path.getsize(npy_path) / (1024**2)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Run benchmarks
    results = {}
    
    if "mahout" in frameworks:
        t_total, throughput, avg_per_sample = run_mahout_numpy(
            num_qubits, num_samples, npy_path
        )
        if throughput > 0:
            results["Mahout"] = {
                "time": t_total,
                "throughput": throughput,
                "avg_per_sample": avg_per_sample,
            }
    
    if "pennylane" in frameworks:
        t_total, throughput, avg_per_sample = run_pennylane_numpy(
            num_qubits, num_samples, npy_path
        )
        if throughput > 0:
            results["PennyLane"] = {
                "time": t_total,
                "throughput": throughput,
                "avg_per_sample": avg_per_sample,
            }
    
    # Print summary
    if results:
        print("\n" + BAR)
        print("SUMMARY")
        print(BAR)
        print(f"{'Framework':<15} {'Time (s)':<12} {'Throughput':<20} {'Avg/Sample':<15}")
        print(SEP)
        
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["throughput"], reverse=True
        )
        
        for name, metrics in sorted_results:
            print(
                f"{name:<15} "
                f"{metrics['time']:<12.4f} "
                f"{metrics['throughput']:<20.1f} "
                f"{metrics['avg_per_sample'] * 1000:<15.2f}"
            )
        
        if len(results) > 1:
            print("\n" + SEP)
            print("SPEEDUP COMPARISON")
            print(SEP)
            
            if "Mahout" in results and "PennyLane" in results:
                speedup = results["Mahout"]["throughput"] / results["PennyLane"]["throughput"]
                print(f"Mahout vs PennyLane: {speedup:.2f}x")
                
                time_ratio = results["PennyLane"]["time"] / results["Mahout"]["time"]
                print(f"Time reduction: {time_ratio:.2f}x faster")
    
    # Cleanup
    if not args.output:
        os.remove(npy_path)
        print(f"\nCleaned up temporary file: {npy_path}")
    
    print("\n" + BAR)
    print("BENCHMARK COMPLETE")
    print(BAR)


if __name__ == "__main__":
    main()
