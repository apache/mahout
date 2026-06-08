#!/usr/bin/env python3
import time
import torch
import numpy as np
from qumat_qdp import QdpEngine

def benchmark_batch_iqp(num_qubits: int, num_samples: int, iters: int = 50):
    # IQP parameters: n + n*(n-1)/2
    data_len = num_qubits + num_qubits * (num_qubits - 1) // 2
    batch_data = np.random.randn(num_samples, data_len).astype(np.float64)
    
    engine = QdpEngine(0)

    # Warmup
    for _ in range(5):
        _ = engine.encode(batch_data, num_qubits, "iqp")
    torch.cuda.synchronize()

    # Benchmark standard IQP batch encoding
    start_time = time.perf_counter()
    for _ in range(iters):
        _ = engine.encode(batch_data, num_qubits, "iqp")
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / iters
    throughput = num_samples / avg_time
    
    return avg_time * 1e6, throughput

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot benchmark.")
        exit(1)
        
    # Typical large-scale workload
    num_qubits = 14
    batch_sizes = [32, 64, 128, 256]
    
    print(f"Benchmarking IQP Batch Throughput (qubits={num_qubits})")
    print("\n## Batch Performance Analysis")
    print("| Batch Size | Latency (us) | Throughput (states/sec) |")
    print("|------------|--------------|-------------------------|")
    
    for bs in batch_sizes:
        latency, tp = benchmark_batch_iqp(num_qubits, bs)
        print(f"| {bs:<10} | {latency:>12.2f} | {tp:>23.2f} |")
