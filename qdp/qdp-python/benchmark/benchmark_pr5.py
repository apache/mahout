#!/usr/bin/env python3
import time
import torch
import numpy as np
from qumat_qdp import QdpEngine

def benchmark_ozaki_engine(num_qubits: int, num_samples: int = 64, iters: int = 50):
    data_len = num_qubits + num_qubits * (num_qubits - 1) // 2
    batch_data = np.random.randn(num_samples, data_len).astype(np.float64)
    
    engine = QdpEngine(0)

    # Warmup
    for _ in range(5):
        _ = engine.encode(batch_data, num_qubits, "iqp")
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(iters):
        _ = engine.encode(batch_data, num_qubits, "iqp")
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) / iters * 1000
    return avg_time_ms

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot benchmark.")
        exit(1)
        
    print(f"Benchmarking IQP Ozaki Implicit Engine (PR5 Accuracy Optimization)")
    print("\n## Ozaki Performance Analysis (Batch Size 64)")
    print("| Qubits | Dimension | Algorithm | Execution Time (ms) |")
    print("|--------|-----------|-----------|---------------------|")
    
    # N=14: Kronecker Decomposition + Ozaki Engine
    latency_14 = benchmark_ozaki_engine(14)
    print(f"| 14     | 16384     | Ozaki TC  | {latency_14:>19.3f} |")
    
    # N=16: Test scaling
    latency_16 = benchmark_ozaki_engine(16)
    print(f"| 16     | 65536     | Ozaki TC  | {latency_16:>19.3f} |")
