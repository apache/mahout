<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements.  See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to You under the Apache License, Version 2.0
  (the "License"); you may not use this file except in compliance with
  the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# `unsafe` audit of the QDP Rust crates

Tracking issue: #1009. This is a snapshot of `qdp-core`, `qdp-kernels`
and `qdp-python` after the first clean-up patches. It records where
`unsafe` lives, why, how much of it is documented, and what is left to
do.

## How to reproduce the numbers

Undocumented blocks are counted by clippy, so the numbers can be
re-checked:

```bash
cd qdp
cargo clippy -p qdp-core -p qdp-kernels -p qdp-python \
  --all-targets --message-format=short -- \
  -W clippy::undocumented_unsafe_blocks \
  2>&1 | grep -c 'missing a safety comment'
```

The per-category split was made by a small script that sorts every
`unsafe` keyword (comments and strings ignored) by what it touches.
Treat the split as approximate. The totals are exact.

Counts are for the stub build (no `nvcc`). In a CUDA build the same
source is compiled; only `cfg(qdp_no_cuda)` stubs drop out.

## Rules this codebase follows

- **Edition 2024.** `unsafe_op_in_unsafe_fn` warns by default, so
  every unsafe operation inside an `unsafe fn` needs its own
  `unsafe { }` block. Such a block in a thin wrapper is **not**
  redundant: removing it is a warning (E0133). Keep the block and add a
  `// SAFETY:` comment that says which contract it relies on.
- **Stubs match the real signature.** No-CUDA stubs for FFI functions
  (`qdp-core/src/gpu/cuda_ffi.rs` and `qdp-kernels/src/lib.rs`) are
  `unsafe fn`, exactly like the `extern "C"` declarations they replace.
  Call sites then need no `#![allow(unused_unsafe)]`.
- **`// SAFETY:` on every block, `# Safety` on every `unsafe fn`.**
  Modules that already follow this switch on
  `clippy::undocumented_unsafe_blocks` locally (see "Lint status").

## Where `unsafe` lives (non-test code)

Keywords counted: `unsafe` blocks, `unsafe fn`, `unsafe impl`,
`unsafe extern` blocks. "Undoc" = blocks/impls clippy flags as lacking a
`SAFETY` comment, before → after this patch series.

Categories:

- **FFI**: CUDA Runtime calls (`cudaMemcpyAsync`, events, ...)
- **Kern**: `qdp_kernels::launch_*` and the GPU validation wrappers
- **Ptr**: raw GPU pointer encode API (`encode_*_from_gpu_ptr*`)
- **DLP**: DLPack lifecycle (deleter, capsules, free)
- **S/S**: `unsafe impl Send/Sync`
- **Misc**: cudarc allocs, slice-from-raw, pointer offsets

### qdp-core

| File | Total | Main kind | Undoc |
|---|---|---|---|
| `lib.rs` | 33 | Ptr | 17 → 0 |
| `dlpack.rs` | 18 | DLP, FFI | 6 → 0 |
| `pipeline_runner.rs` | 6 | DLP | 5 |
| `encoding/*.rs` | 9 | Kern, Misc | 9 |
| `gpu/cuda_ffi.rs` | 17 | FFI | 0 |
| `gpu/cuda_sync.rs` | 1 | FFI | 1 |
| `gpu/memory.rs` | 15 | Misc, FFI, S/S | 15 → 14 |
| `gpu/metrics.rs` | 2 | FFI | 2 |
| `gpu/overlap_tracker.rs` | 7 | FFI | 6 |
| `gpu/pipeline.rs` | 13 | FFI | 9 |
| `gpu/validation.rs` | 8 | Kern | 4 |
| `gpu/encodings/amplitude.rs` | 35 | Kern, Ptr | 24 |
| `gpu/encodings/angle.rs` | 23 | Kern, Ptr | 17 |
| `gpu/encodings/basis.rs` | 18 | Kern, Ptr | 12 |
| `gpu/encodings/iqp.rs` | 6 | Kern | 4 |
| `gpu/encodings/phase.rs` | 9 | Kern | 7 |
| `gpu/encodings/mod.rs` | 4 | Ptr (trait) | 0 |
| `readers/tensorflow.rs` | 1 | Misc | 0 |

`dlpack.rs` went from 10 to 18 keywords because the two
`#[allow(unsafe_op_in_unsafe_fn)]` functions now spell out each unsafe
operation in its own commented block. That is more `unsafe` keywords
but less unchecked code.

### qdp-kernels

| File | Total | Main kind | Undoc |
|---|---|---|---|
| `src/lib.rs` | 31 | Kern | 4 |

1 `unsafe extern "C"` block (26 launchers), 26 `unsafe` stubs, and 4
`unsafe impl` of cudarc's `DeviceRepr` / `ValidAsZeroBits` for the
`#[repr(C)]` complex types (the 4 undocumented items).

### qdp-python

| File | Total | Main kind | Undoc |
|---|---|---|---|
| `src/dlpack.rs` | 11 | DLP | 7 |
| `src/engine.rs` | 2 | Ptr | 2 |
| `src/tensor.rs` | 6 | DLP, S/S | 5 |

### Totals

| Scope | Before | After |
|---|---|---|
| Non-test keywords | 243 | 275 |
| Non-test undoc | 156 | 132 |
| Tests, examples, build.rs undoc | 201 | 201 |
| `allow(unused_unsafe)` | 13 | 0 |
| `allow(unsafe_op_in_unsafe_fn)` | 2 | 0 |
| `unsafe impl Send/Sync` | 6 | 4 |

The keyword count goes *up* (26 stubs became `unsafe fn`, 8 explicit
blocks in `dlpack.rs`). That is expected: the goal of this round is
fewer *unexplained* unsafe operations and no blanket allows, not a
smaller grep count.

Test and example code (`qdp-core/tests`, `qdp-kernels/tests`,
`qdp-core/examples`) holds 208 more keywords, almost all
`unsafe { engine.encode_*_from_gpu_ptr*(...) }` and
`unsafe { free_dlpack_tensor(ptr) }`. They are left undocumented on
purpose. The roadmap items below remove most of them by making those
APIs safe.

## What was done (by commit)

1. **dfe118e (#1011, @0lai0, already on `main`).** Added
   `dlpack::free_dlpack_tensor`, a checked helper that `take()`s the
   deleter so it cannot run twice, and moved the examples to it.
2. **No-CUDA kernel stubs are now `unsafe fn`.** This removed all 13
   `#![allow(unused_unsafe)]` attributes. No allow was needed in the
   end, so there is no "scoped allow" to document.
3. **`dlpack.rs` fully documented.** Each block has a `SAFETY` comment,
   the two `allow(unsafe_op_in_unsafe_fn)` are gone, and
   `clippy::undocumented_unsafe_blocks` is on for the module.
4. **`QdpEngine` pointer API documented.** 17 blocks commented, and the
   lint is on for `impl QdpEngine`.
5. **`# Safety` sections for 5 crate-private `unsafe fn`s** that lacked
   them. All public ones already had one.
6. **`GpuStateVector` Send/Sync.** The manual `unsafe impl`s are
   replaced by a compile-time assertion. The type is auto-Send/Sync.

## Known gaps worth a closer look

These are notes, not confirmed bugs.

- `qdp-python/src/tensor.rs`: `unsafe impl Send/Sync for QuantumTensor`
  says the pointer is "thread-safe via Arc in the deleter". The
  `consumed` flag and the raw `DLManagedTensor` are not synchronised.
  pyo3 normally guards access with the GIL, but the reasoning should
  be written down (or the type marked `#[pyclass(unsendable)]`).
- `gpu/memory.rs` `PinnedHostBuffer`: the `unsafe impl Send/Sync` is
  needed (it owns a raw host pointer), but `as_slice_mut` hands out
  `&mut [T]` over memory an in-flight `cudaMemcpyAsync` may still read.
  The pipeline avoids this by syncing before reuse; that rule should
  be a `# Safety` note or encoded in the type.
- `dlpack_deleter` assumes `ndim` matches the length of the boxed
  `shape`/`strides` slices. True for tensors made by `to_dlpack`
  (always 2-D). A tensor from another producer must never reach it.
- Many `launch_*` call sites compute `state_ptr.add(offset)` inside
  `unsafe`. The bounds come from loop arithmetic that is easy to check
  but nowhere written down.

## Roadmap for the rest of #1009

Ordered by value per review effort. Each step is one small PR.

1. **Document the encoder modules.** `gpu/encodings/*.rs` holds 64 of
   the 132 remaining undocumented blocks. Nearly all are
   `launch_*` calls with the same argument: "pointers come from a
   `CudaSlice` / validated input of the stated length; `stream` is
   the caller's". A shared comment template keeps it short. Then turn
   on the lint for `gpu::encodings`.
2. **Document `gpu/pipeline.rs`, `overlap_tracker.rs`,
   `memory.rs`.** Mostly CUDA event/stream handles owned by the
   struct. Write down the `PinnedHostBuffer` rule above.
3. **One FFI module.** `cuda_ffi.rs` already centralises the CUDA
   Runtime, but `metrics.rs` calls `cudarc::driver::sys::lib()`
   directly and `qdp-kernels` launchers are called from ~10 files.
   Add thin *safe* wrappers in one `gpu::ffi` module (e.g.
   `fn launch_l2_norm(dev: &CudaSlice<f64>, out: &mut CudaSlice<f64>,
   stream: &CudaStream) -> Result<()>`). The `unsafe` then lives in
   one file, and slice lengths are checked once, by type.
4. **DLPack RAII owner.** dfe118e added `free_dlpack_tensor`. The next
   step from the issue is a small owner type:
   ```rust
   pub struct DlpackOwned(NonNull<DLManagedTensor>);
   impl Drop for DlpackOwned { /* take deleter, call it */ }
   impl DlpackOwned { pub fn into_raw(self) -> *mut _ }
   ```
   `QdpEngine::encode*` could return it (keeping a `*_raw` variant for
   FFI). Tests, examples, `pipeline_runner::release_dlpack`, and
   qdp-python's `QuantumTensor::drop` / `DLPackTensorInfo::drop` would
   all stop calling deleters by hand. That removes most of the ~60
   DLPack-cleanup blocks in tests.
5. **Validated GPU pointer handle.** The issue's main proposal:
   ```rust
   pub struct GpuInput<'a, T> {
       ptr: NonNull<T>,
       len: usize,
       device: i32,
       _lt: PhantomData<&'a T>,
   }
   ```
   Built by one `unsafe fn GpuInput::from_raw(ptr, len, &device)` that
   runs today's `validate_cuda_input_ptr` checks. Then
   `QdpEngine::encode_from(&self, input: GpuInput<'_, f32>, ...)` is a
   **safe** fn, and the eight `encode_*_from_gpu_ptr*` pairs collapse
   to one generic entry point. Callers make exactly one `unsafe` call,
   at the boundary where the pointer enters Rust (in qdp-python, from
   the DLPack capsule). This removes most of the 75 pointer-API blocks
   in `gpu_ptr_encoding.rs`. It is a public API change, so it needs a
   deprecation cycle for the old methods.
6. **Crate-wide lint.** After steps 1 to 3, enable
   `#![warn(clippy::undocumented_unsafe_blocks)]` for all of
   `qdp-core` and add `clippy::multiple_unsafe_ops_per_block` where
   it does not explode.
