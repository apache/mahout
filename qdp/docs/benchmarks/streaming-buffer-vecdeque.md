# StreamingProducer buffer: `Vec` + cursor vs `VecDeque`

Before/after benchmark for the change in [#1462](https://github.com/apache/mahout/pull/1462),
which replaced the `StreamingProducer` read buffer — a `Vec<T>` plus a read cursor — with a
`VecDeque<T>` ring buffer.

## What changed

`StreamingProducer::produce` is the streaming hot path: every batch refills the buffer from the
Parquet reader, copies one batch out, and discards the consumed prefix.

- **Before.** A `Vec` plus a `buffer_cursor`. The consumed prefix was reclaimed with
  `buffer.drain(..buffer_cursor)` once the cursor passed the half-way mark, and that `drain`
  memmoves every element still retained past the cursor.
- **After.** A `VecDeque`. `drain(..take)` on a front-anchored range only advances the ring's
  head, so nothing retained is moved. The batch copy is stitched from the ring's two
  `as_slices()` halves to stay on `extend_from_slice`'s bulk path.

## Running it

```bash
make -C qdp bench_streaming_buffer

# heavier / lighter workload
BENCH_REPEATS=15 BENCH_BATCHES=500 make -C qdp bench_streaming_buffer
```

Both benchmarks live in `qdp-core/src/pipeline_runner.rs` under `mod tests::buffer_bench` and are
`#[ignore]`d, so `make test_rust` never pays for them. They must be in-tree rather than under
`examples/` because `pipeline_runner` is a private module — that is also what lets the "after"
side be the shipped `StreamingProducer` itself instead of a copy of it. The "before" side is a
frozen, verbatim copy of the pre-#1462 producer (`LegacyStreamingProducer`); it is the benchmark's
baseline and must not be updated to track the current implementation.

Each side is run interleaved, best-of-N, with only the `produce()` calls timed and each batch's
buffer recycled into the next call exactly as `spawn_producer` does. In the end-to-end benchmark
both sides are additionally asserted to emit a checksum-identical element stream, so it doubles as
an equivalence check between the old and new producer.

## Results

Intel Xeon w3-2435 (16 threads), 30 GB RAM, rustc 1.94.0, `--release`, `BENCH_REPEATS=15`.

### Buffer management in isolation

The reader is replaced by a pre-filled slice, so this is the buffer work alone. The refill chunk
is fixed at 65536 elements — the `INITIAL_CHUNK_CAP` that `build_streaming_producer` uses.

| batch elements | batch:chunk | before (Vec+cursor) | after (VecDeque) | speedup |
|---|---|---|---|---|
| 4096 | 1:16 | 782 ns/batch | 504 ns/batch | **1.55x** |
| 16384 | 1:4 | 3370 ns/batch | 2480 ns/batch | **1.36x** |
| 65536 | 1:1 | 9668 ns/batch | 9716 ns/batch | 1.00x |
| 262144 | 4:1 | 54176 ns/batch | 54469 ns/batch | 0.99x |

The ratio of batch size to refill chunk is what drives the difference, not absolute size. A
legacy compaction memmoves whatever is retained past the cursor: the smaller a batch is relative
to a refill chunk, the more is left behind at each compaction and the more the old code moved. At
1:1 and above, a compaction leaves nothing behind, the memmove is empty, and the two strategies do
identical work — which is why the change is neutral, not negative, at large batch sizes.

### End-to-end through `produce()`

Same producers, but reading a real Parquet file through `ParquetStreamingReader`.

| shape | batches | before (Vec+cursor) | after (VecDeque) | speedup |
|---|---|---|---|---|
| 8 qubits (256 x 16) | 400 | 7.23 ms (18.1 us/batch) | 7.12 ms (17.8 us/batch) | 1.02x |
| 8 qubits (256 x 64) | 200 | 18.40 ms (92.0 us/batch) | 18.24 ms (91.2 us/batch) | 1.01x |
| 10 qubits (1024 x 64) | 100 | 33.17 ms (331.7 us/batch) | 32.76 ms (327.6 us/batch) | 1.01x |
| 12 qubits (4096 x 32) | 60 | 3.41 ms (56.9 us/batch) | 3.34 ms (55.6 us/batch) | 1.02x |

## Interpretation

Read the two tables together, and they agree. In the most favourable realistic shape (8 qubits,
batch 16 — a 1:16 batch-to-chunk ratio) the isolated benchmark puts the saving at ~278 ns/batch,
against an end-to-end `produce()` cost of ~18.1 us/batch. That predicts a ~1.5% end-to-end gain,
and 1.02x is what the end-to-end table measures.

So, stated plainly: **Parquet decode is ~98% of `produce()`, and this change makes the remaining
~2% about 1.4–1.6x cheaper in the regimes where it does anything at all.** The end-to-end effect
is real but small, and at the sizes tested it sits close to run-to-run noise. The change should be
justified as removing an O(retained) memmove from the hot path — bounding worst-case per-batch
cost and keeping buffer capacity constant — rather than as a headline throughput win.

Two properties the timings do not capture, both covered by
`test_streaming_producer_buffer_capacity_constant_over_100_batches`:

- Buffer capacity is now provably constant across a long stream; the old code's compaction
  threshold made peak capacity depend on the arrival pattern.
- `build_streaming_producer` pre-reserves the steady-state peak (`required + initial_cap`), so
  the ring never reallocates mid-run, from batch 0 onward.
