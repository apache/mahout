# QDP Runtime v1

## Summary

`qdp-runtime` is the v1 control-plane and execution skeleton for state-partitioned
distributed execution on top of Mahout QDP.

The current v1 implementation focuses on:

- state-partitioned metadata
- worker registration and device inventory
- weighted and topology-aware placement
- partition task generation and lifecycle
- in-process execution loops
- gather and metric-reduction planning
- a minimal runtime object/output model

This is intentionally a minimal v1. It is not yet a full multi-node transport
layer or a persistent GPU object store.

## Current Object and Output Model

Runtime outputs are tracked as `RuntimeObjectRecord`s inside the coordinator.

Current object kinds:

- `EncodedPartition`
- `ReducedMetric`

Each runtime object records:

- `object_id`
- `job_id`
- `partition_id`
- `kind`
- `location`
- `handle`
- `ready`

In v1, the object model is metadata-first. It gives the runtime a stable way to
track outputs across task completion, gather planning, and reduce planning,
without requiring a full persistent object store yet.

## Manual End-to-End Path

There are two example entry points in `qdp-runtime/examples`:

- `local_runtime_smoke.rs`
- `local_runtime_benchmark.rs`

### Smoke Example

This runs:

1. worker registration
2. job planning
3. in-process task execution
4. object registration
5. gather plan construction

Run it with:

```bash
cd qdp
cargo run -p qdp-runtime --example local_runtime_smoke
```

### Minimal Benchmark

This prints basic timings for:

- planning
- execution
- partition/object counts

Run it with:

```bash
cd qdp
cargo run -p qdp-runtime --example local_runtime_benchmark
```

## Optional Local QDP Integration

When built with the `local-executor` feature, `qdp-runtime` can use
`LocalEncodeWorkerExecutor` to call the real `qdp-core` encode path from the
runtime.

Example command:

```bash
cd qdp
cargo test -p qdp-runtime --features local-executor
```

## What v1 Supports

- `PartitionLocalConsume` metadata
- `GatherFullState` planning
- `ReduceMetrics` planning
- retry and lease timeout skeleton
- NVLink-aware placement hints

## What v1 Does Not Yet Support

- cross-node transport
- persistent GPU object store
- partition migration
- dynamic repartitioning
- full collective communication
- full benchmark suite integration
