
import numpy as np
import time
import os
import sys

# Add the current directory to sys.path to ensure we can import the locally built qumat_qdp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qumat_qdp import QdpEngine

def benchmark_iqp_tc():
    print("=== IQP Tensor Core Acceleration Benchmark ===")
    
    # Test across different qubit counts
    for num_qubits in [8, 10, 12, 14, 16]:
        batch_size = 1024
        num_params = num_qubits + num_qubits * (num_qubits - 1) // 2
        
        print(f"\nConfig: num_qubits={num_qubits}, batch_size={batch_size}")
        
        data = np.random.randn(batch_size, num_params).astype(np.float64)
        
        try:
            engine = QdpEngine(0, precision="float64")
        except Exception as e:
            print(f"Failed to initialize QdpEngine: {e}")
            return

        # Warmup
        for _ in range(5):
            _ = engine.encode(data, num_qubits, "iqp")
            if hasattr(engine, "encode_batch_tc"):
                _ = engine.encode_batch_tc(data, num_qubits)
        
        # Benchmark FWT (current encode_batch)
        start = time.time()
        iters = 50
        for _ in range(iters):
            _ = engine.encode(data, num_qubits, "iqp")
        end = time.time()
        fwt_time = (end - start) / iters
        print(f"  FWT Avg Time:         {fwt_time*1000:8.3f} ms")
        
        # Benchmark TC
        if hasattr(engine, "encode_batch_tc"):
            start = time.time()
            for _ in range(iters):
                _ = engine.encode_batch_tc(data, num_qubits)
            end = time.time()
            tc_time = (end - start) / iters
            print(f"  Tensor Core Avg Time: {tc_time*1000:8.3f} ms")
            
            speedup = fwt_time / tc_time
            print(f"  Speedup:              {speedup:8.2f}x")
        else:
            print("  Tensor Core (encode_batch_tc) not available in this build.")

if __name__ == "__main__":
    benchmark_iqp_tc()
