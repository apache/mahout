# Observability Tools Usage Guide

This document describes how to use the observability tools to diagnose pipeline performance and verify optimization targets.

**Note for Python Users**: The Python bindings automatically initialize Rust logging when the module is imported. You only need to set the `RUST_LOG` environment variable before importing `qumat.qdp`. See the [Logging Configuration](#logging-configuration) section for details.

## Table of Contents

- [Overview](#overview)
- [Enabling Observability](#enabling-observability)
- [Logging Configuration](#logging-configuration)
- [Pool Metrics](#pool-metrics)
- [Overlap Tracking](#overlap-tracking)
- [Performance Impact](#performance-impact)
- [Usage Examples](#usage-examples)
- [Integration with Benchmarking](#integration-with-benchmarking)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Related Documentation](#related-documentation)

## Overview

The observability tools provide two main features:

1. **Pool Metrics**: Track pinned buffer pool utilization to diagnose pool starvation
2. **Overlap Tracking**: Measure H2D copy and compute overlap to verify >60% overlap target

Both features are **optional** and can be enabled via environment variables. When disabled, they have **zero performance overhead**.

**Important**: Observability features are only active when the dual-stream pipeline is used. The pipeline is automatically used for data sizes >= 1MB (131,072 elements for float64, which corresponds to 17 qubits or more).

## Enabling Observability

### Pool Metrics

Enable pool utilization metrics by setting the `QDP_ENABLE_POOL_METRICS` environment variable:

```bash
export QDP_ENABLE_POOL_METRICS=1
# or
export QDP_ENABLE_POOL_METRICS=true
```

### Overlap Tracking

Enable H2D overlap tracking by setting the `QDP_ENABLE_OVERLAP_TRACKING` environment variable:

```bash
export QDP_ENABLE_OVERLAP_TRACKING=1
# or
export QDP_ENABLE_OVERLAP_TRACKING=true
```

### Both Features

You can enable both features simultaneously:

```bash
export QDP_ENABLE_POOL_METRICS=1
export QDP_ENABLE_OVERLAP_TRACKING=1
```

## Logging Configuration

The observability tools use the Rust `log` crate. To see the output, you need to configure a logger. The Python bindings automatically initialize `env_logger` when the module is imported, so you only need to set the `RUST_LOG` environment variable.

### Using Python Bindings (Recommended for Python Users)

The Python bindings (`qumat.qdp`) automatically initialize the Rust logging system when the module is imported. Simply set the `RUST_LOG` environment variable before importing:

```python
import os

# Set environment variables BEFORE importing qumat.qdp
os.environ['RUST_LOG'] = 'info'  # or 'debug' for more detailed output
os.environ['QDP_ENABLE_POOL_METRICS'] = '1'
os.environ['QDP_ENABLE_OVERLAP_TRACKING'] = '1'

# Now import - logging will be initialized automatically
from qumat import qdp
import numpy as np

# Your QDP code here
# Note: Use >= 18 qubits (2MB) to ensure pipeline is triggered
engine = qdp.QdpEngine(0)
data = np.random.rand(262144).astype(np.float64)  # 18 qubits = 262144 elements = 2MB
data = data / np.linalg.norm(data)
result = engine.encode(data, num_qubits=18, encoding_method='amplitude')
```

Or set environment variables in your shell:

```bash
export RUST_LOG=info
export QDP_ENABLE_POOL_METRICS=1
export QDP_ENABLE_OVERLAP_TRACKING=1

python your_script.py
```

**Note**: The Python `logging` module is separate from Rust's logging system. Rust log messages will appear in stderr, not through Python's logging system.

### Using Rust Examples

For Rust examples, you need to manually initialize `env_logger`:

```rust
// In your example main function:
env_logger::Builder::from_default_env()
    .init();  // Don't override filter_level - let RUST_LOG control it
```

Then run with logging enabled:

```bash
RUST_LOG=info cargo run --example observability_test --release
```

## Pool Metrics

### What It Tracks

- **min_available**: Minimum number of buffers available during any acquire operation
- **max_available**: Maximum number of buffers available during any acquire operation
- **total_acquires**: Total number of buffer acquire operations
- **total_waits**: Number of times a thread had to wait because the pool was empty
- **starvation_ratio**: Ratio of waits to acquires (indicates pool starvation)
- **avg_wait_time_ns**: Average wait time in nanoseconds

### Interpreting Results

**Good Performance**:
- `starvation_ratio < 0.05` (less than 5% of acquires had to wait)
- `avg_wait_time_ns` is small (< 1ms typically)

**Pool Starvation Detected**:
- `starvation_ratio > 0.05` (more than 5% of acquires had to wait)
- The tool will automatically log a warning
- **Action**: Consider increasing `QDP_PINNED_POOL_SIZE` (when implemented in future PR)

### Example Output

```
[INFO] Pool Utilization: min=0, max=2, acquires=100, waits=2, starvation=2.00%
```

If starvation is detected:

```
[WARN] Pool starvation detected: 2.0% of acquires had to wait. Consider increasing pool size.
```

## Overlap Tracking

### What It Tracks

- **H2D Overlap Ratio**: Percentage of time that copy and compute operations overlap
- Measured per chunk (logged for chunk 0 and every 10th chunk to avoid excessive output)

### Interpreting Results

**Target**: H2D overlap should be **>60%** for optimal performance.

**Good Performance**:
- Overlap > 60%: Pipeline is efficiently overlapping copy and compute
- Overlap percentage is logged at `INFO` level
- Detailed timing information (copy time, compute time, overlap time) is logged at `DEBUG` level

**Below Target**:
- Overlap < 60%: Pipeline is not achieving optimal overlap
- Overlap percentage is logged at `INFO` level
- A warning message is logged at `WARN` level
- Detailed timing information is available at `DEBUG` level for troubleshooting
- **Possible causes**:
  - Regular synchronization points (will be addressed in future PR)
  - Pool size too small
  - Chunk size not optimal for hardware

### Example Output

**Good Overlap**:
```
[INFO] Chunk 0: H2D overlap = 68.5%
[INFO] Chunk 10: H2D overlap = 72.3%
```

**Below Target**:
```
[INFO] Chunk 0: H2D overlap = 45.2%
[WARN] Chunk 0: Overlap below target (60%), current = 45.2%
```

**With DEBUG level** (shows detailed timing information):
```
[INFO] Chunk 0: H2D overlap = 68.5%
[DEBUG] Chunk 0: H2D overlap details - copy=1.230ms, compute=2.450ms, overlap=0.840ms, ratio=68.5%
```

## Performance Impact

### When Disabled

- **Zero overhead**: All observability code is conditionally compiled or uses fast checks
- No CUDA events created
- No atomic operations performed
- Safe to leave in production code

### When Enabled

- **Pool Metrics**: < 1% CPU overhead (atomic operations with Relaxed ordering)
- **Overlap Tracking**: < 5% CPU overhead in debug mode (CUDA event queries)
- **Combined**: < 5% CPU overhead when both enabled

## Usage Examples

### Example 1: Using the Rust Test Example

A test example is provided to demonstrate observability features:

```bash
# Build the example
# Note: Run from the qdp directory (or use: cd qdp)
cargo build -p qdp-core --example observability_test --release

# Run without observability (baseline)
./target/release/examples/observability_test

# Run with pool metrics only
QDP_ENABLE_POOL_METRICS=1 RUST_LOG=info ./target/release/examples/observability_test

# Run with overlap tracking only
QDP_ENABLE_OVERLAP_TRACKING=1 RUST_LOG=info ./target/release/examples/observability_test

# Run with both features
QDP_ENABLE_POOL_METRICS=1 QDP_ENABLE_OVERLAP_TRACKING=1 RUST_LOG=info ./target/release/examples/observability_test
```

### Example 2: Diagnose Pool Starvation (Python)

```python
import os
os.environ['QDP_ENABLE_POOL_METRICS'] = '1'
os.environ['RUST_LOG'] = 'info'

from qumat import qdp
import numpy as np

engine = qdp.QdpEngine(0)
# Create data that triggers pipeline (>= 1MB = 131072 elements = 17 qubits)
# Using 18 qubits (262144 elements = 2MB) to ensure pipeline is used
data = np.random.rand(262144).astype(np.float64)
data = data / np.linalg.norm(data)
result = engine.encode(data, num_qubits=18, encoding_method='amplitude')
# Check stderr for starvation warnings
```

Or using shell environment variables:

```bash
# Enable pool metrics
export QDP_ENABLE_POOL_METRICS=1
export RUST_LOG=info

# Run your workload
python your_script.py

# Check stderr output for starvation warnings
```

### Example 3: Verify Overlap Target (Python)

```python
import os
os.environ['QDP_ENABLE_OVERLAP_TRACKING'] = '1'
os.environ['RUST_LOG'] = 'info'  # or 'debug' for detailed timing

from qumat import qdp
import numpy as np

engine = qdp.QdpEngine(0)
# Create data that triggers pipeline (>= 1MB = 131072 elements = 17 qubits)
# Using 18 qubits (262144 elements = 2MB) to ensure pipeline is used
data = np.random.rand(262144).astype(np.float64)
data = data / np.linalg.norm(data)
result = engine.encode(data, num_qubits=18, encoding_method='amplitude')
# Check stderr for overlap percentages
```

Or using shell environment variables:

```bash
# Enable overlap tracking
export QDP_ENABLE_OVERLAP_TRACKING=1
export RUST_LOG=info  # or debug for detailed timing

# Run your workload
python your_script.py

# Check stderr output for overlap percentages
```

### Example 4: Full Observability (Python)

```python
import os
os.environ['QDP_ENABLE_POOL_METRICS'] = '1'
os.environ['QDP_ENABLE_OVERLAP_TRACKING'] = '1'
os.environ['RUST_LOG'] = 'info'  # or 'debug' for more details

from qumat import qdp
import numpy as np

engine = qdp.QdpEngine(0)
# Create data that triggers pipeline (>= 1MB = 131072 elements = 17 qubits)
# Using 18 qubits (262144 elements = 2MB) to ensure pipeline is used
data = np.random.rand(262144).astype(np.float64)
data = data / np.linalg.norm(data)
result = engine.encode(data, num_qubits=18, encoding_method='amplitude')
# Check stderr for both metrics
```

Or using shell environment variables:

```bash
# Enable both features
export QDP_ENABLE_POOL_METRICS=1
export QDP_ENABLE_OVERLAP_TRACKING=1
export RUST_LOG=info  # or debug for more details

# Run your workload
python your_script.py

# Check stderr output for both pool utilization and overlap
```

## Integration with Benchmarking

When running benchmarks, enable observability to collect data:

**Python Benchmarks**:
```bash
# Enable observability
export QDP_ENABLE_POOL_METRICS=1
export QDP_ENABLE_OVERLAP_TRACKING=1
export RUST_LOG=info

# Run throughput benchmark (example - adjust path and arguments as needed)
python benchmark_throughput.py --qubits 16 --batches 200 --batch-size 64 --frameworks mahout

# Check stderr output for:
# - Pool utilization summary
# - Overlap percentages
# - Any warnings about starvation or low overlap
```

**Rust Benchmarks**:
```bash
# Enable observability
export QDP_ENABLE_POOL_METRICS=1
export QDP_ENABLE_OVERLAP_TRACKING=1
export RUST_LOG=info

# Run benchmark (ensure env_logger is initialized in your benchmark code)
# Replace 'your_benchmark' with the actual benchmark binary name
cargo run --release --bin your_benchmark
```

## Troubleshooting

### No Log Output

**Problem**: Enabled observability but see no log output.

**For Python Users**:
1. **Set `RUST_LOG` environment variable** before importing `qumat.qdp`:
   ```python
   import os
   os.environ['RUST_LOG'] = 'info'  # Must be set BEFORE import
   from qumat import qdp
   ```
   Or set it in your shell:
   ```bash
   export RUST_LOG=info
   python your_script.py
   ```
2. **Check stderr, not stdout**: Rust log messages go to stderr, not stdout
3. **Verify environment variables are set**: Check that `QDP_ENABLE_POOL_METRICS` and/or `QDP_ENABLE_OVERLAP_TRACKING` are set to `1` or `true`
4. **Ensure pipeline is used**: Observability is only active when the dual-stream pipeline is used (data size >= 1MB threshold, which is 131,072 elements for float64, corresponding to 17 qubits or more)

**For Rust Users**:
1. Ensure `env_logger` is initialized in your code:
   ```rust
   env_logger::Builder::from_default_env().init();
   ```
2. Set `RUST_LOG=debug` or `RUST_LOG=info` when running
3. Check that the code path actually uses the pipeline (observability is only active in `run_dual_stream_pipeline`)

### Overlap Always 0%

**Problem**: Overlap tracking shows 0% overlap.

**Possible Causes**:
1. Events not being recorded (check that `record_copy_start/end` and `record_compute_start/end` are called)
2. Events not completing (check for CUDA errors)
3. Compute and copy not actually overlapping (expected in some scenarios)

**Debug Steps**:
1. Enable debug logging: `RUST_LOG=debug`
2. Check for CUDA errors in the logs
3. Verify that both copy and compute operations are actually running

### High Starvation Ratio

**Problem**: Pool starvation ratio > 5%.

**Solutions**:
1. **Immediate**: This indicates the pool size may be too small
2. **Future**: When `QDP_PINNED_POOL_SIZE` environment variable is implemented, increase it
3. **Workaround**: For now, this is expected behavior with the current fixed pool size of 2

## API Reference

### PoolMetrics

```rust
use qdp_core::gpu::PoolMetrics;

// Create metrics
let metrics = PoolMetrics::new();

// Record operations (automatically done by PinnedBufferPool when enabled)
// ...

// Generate report
let report = metrics.report();
report.print_summary();

// Reset for new measurement period
metrics.reset();
```

### OverlapTracker

```rust
use qdp_core::gpu::OverlapTracker;

// Create tracker (usually done automatically by pipeline)
let tracker = OverlapTracker::new(pool_size, enabled)?;

// Record events (automatically done by pipeline when enabled)
tracker.record_copy_start(stream, slot)?;
tracker.record_copy_end(stream, slot)?;
tracker.record_compute_start(stream, slot)?;
tracker.record_compute_end(stream, slot)?;

// Calculate and log overlap
tracker.log_overlap(chunk_idx)?;
```

## Related Documentation

- [NVTX Usage Guide](./NVTX_USAGE.md) - NVTX profiling markers

## Future Enhancements

Planned improvements in future PRs:

1. **Dynamic Pool Size**: `QDP_PINNED_POOL_SIZE` environment variable to adjust pool size
2. **More Detailed Metrics**: Per-chunk timing breakdowns
3. **Export to Metrics Format**: Export metrics to Prometheus/StatsD format
4. **Real-time Monitoring**: Web dashboard for live metrics
