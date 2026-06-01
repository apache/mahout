---
title: Introducing Apache Mahout qumat 0.6.0
date: 2026-06-01
tags: [release, qumat, qdp]
authors: [ryankert, mahout-team]
---

We're excited to announce the release of **Apache Mahout qumat 0.6.0**, rolling up 111 pull requests from contributors across the community since 0.5.0. This release brings AMD GPU support to first-class status, closes encoder parity gaps between CUDA and ROCm, adds new benchmarks for real-world datasets, and overhauls the documentation site.

Thank you to everyone who contributed to this release: Ryan Huang, Vic Wen, Jie-Kai Chang, Guan-Ming (Wesley) Chiu, Tim Hsiung, Kuan-Hao Huang, ChenChen Lai, Suyash Parmar, Yehfela, Shivam Mittal, Alisha, Hsien-Cheng Huang, Eddie Tsai, Andrew Musselman, Xin Hao, wdskuki, Trevor Grant, Karanjot Gaidu, Howardisme, and Han-Wen Tsao.

<!-- truncate -->

## Key Highlights

- **QDP encoding parity and new encodings** — Phase, IQP, and IQP-Z encodings on both CUDA and AMD ROCm; float32 zero-copy batch paths and GPU-pointer encoding for the IQP family.
- **AMD GPU support** — New Mahout-AMD framework with hand-written Triton kernels running natively on ROCm; AMD is now selectable from all QDP encoding and throughput benchmarks.
- **New benchmarks** — SVHN Quantum Kernel SVM, MNIST amplitude encoding, and IQP latency/throughput benchmarks with PennyLane baselines.
- **Documentation overhaul** — Frontmatter across all pages, self-hosted KaTeX, a new troubleshooting guide, and CONTRIBUTING merged into the README for easier onboarding.

Let's explore these in more detail.

## QDP Encoding Parity and New Encodings

Prior to 0.6.0, the CUDA and AMD ROCm backends had different encoder coverage — some encoding paths only existed on one backend, which made benchmarking apples-to-apples comparisons difficult and limited ROCm's usability in practice.

In 0.6.0, Phase, IQP, and IQP-Z encodings now ship on both NVIDIA CUDA and AMD ROCm backends, closing the parity gap with the existing angle and amplitude paths.

On the performance side, this release adds:
- **Float32 zero-copy batch paths** for angle and basis encoders — both single-sample and batched, in both `qdp-core` (Rust) and the Python bindings — eliminating host-device copies in hot loops.
- **GPU-pointer encoding** for the IQP family — pass a CUDA tensor directly via DLPack and skip the host round-trip entirely.
- **IQP kernel fusion and grid-stride optimizations** — persistent kernels and fused encode passes reduce kernel launch overhead.
- **Async prefetching and native f32 dispatch pipelines** — overlaps I/O with compute for throughput-bound workloads.

![Data-to-state latency scaling (ms/vector, log scale) for amplitude, angle, and basis encoding. QDP (orange) achieves orders-of-magnitude lower latency than both CPU and GPU backends.](./encoding-latency-scaling.png)

*Figure: Data-to-state latency scaling (ms/vector, log scale) for amplitude, angle, and basis encoding. QDP (orange) achieves orders-of-magnitude lower latency than both CPU and GPU backends.*

Install the QDP extra to try these:

```bash
pip install "qumat[qdp]==0.6.0"  # Linux x86_64 + NVIDIA CUDA
```

## AMD GPU Support

0.6.0 makes AMD ROCm a first-class backend via **hand-written Triton kernels** (`TritonAmdEngine`) that run natively on ROCm without going through PennyLane. The Triton path covers amplitude, angle, basis, phase, IQP, and IQP-Z encodings — the same set now available on CUDA — and is selectable from the QDP encoding and throughput benchmarks via `--qdp-backend amd`.

PennyLane-AMDGPU (`lightning.amdgpu`) appears in the throughput benchmark as a comparison baseline, not as Mahout's implementation path.

Docker images for the AMD environment are also included (`Dockerfile.qdp-amd`), making it easy to spin up a reproducible AMD test environment without a physical ROCm machine.

CUDA kernel build targets are now **configurable** for forward compatibility — specify the target compute capabilities at build time rather than hard-coding them, so the wheel stays valid on architectures released after the build.

## New Benchmarks

0.6.0 adds three new benchmarks that cover more realistic workloads than the synthetic microbenchmarks from earlier releases:

**SVHN IQP variational classifier** — trains a variational IQP classifier on the Street View House Numbers dataset (digit 1 vs 7, 200 samples, 200 iterations) on two RTX 3090 Ti GPUs. QDP offloads the IQP encoding step to the GPU in a single one-shot pass before training begins, keeping the training backend independent of encoding cost. QDP GPU consistently outperforms PennyLane GPU at every qubit count — **41 vs 33 samples/s at 4 qubits** and **20 vs 15 samples/s at 10 qubits** (~1.35×).

![SVHN IQP training throughput: QDP GPU vs PennyLane GPU at 4–10 qubits (RTX 3090 Ti)](./svhn-benchmark.png)

**SVHN Quantum Kernel SVM** — runs a precomputed squared inner-product quantum kernel SVM on SVHN amplitude-encoded features (12 qubits), measuring end-to-end pipeline time from raw pixels to SVM prediction. This is the first benchmark that exercises the full pipeline (feature encoding → kernel matrix → SVM fit/predict) rather than just the encoding step.

**Data-to-State latency** — isolates the full pipeline from CPU RAM to GPU-ready quantum state at 16 qubits (65,536-dimensional vectors). Mahout (QDP) delivers **0.160 ms/vector** — **4.5× faster than PennyLane** (0.716 ms), **56× faster than Qiskit Statevec** (9.030 ms), and **477× faster than Qiskit Initialize** (76.243 ms). PennyLane and Mahout both target GPU; Qiskit runs on CPU with Qiskit Init adding full circuit decomposition and transpilation overhead on top.

![Data-to-state latency: Mahout 0.160 ms, PennyLane 0.716 ms, Qiskit Statevec 9.030 ms, Qiskit Init 76.243 ms (log scale)](./iqp-benchmark.png)

**DataLoader amplitude throughput** — streams 12,800 amplitude-encoded vectors in batches of 64 at 16 qubits. Mahout (QDP) delivers **6,101 vectors/s** — **3.8× faster than PennyLane** (1,604 vectors/s).

![Amplitude DataLoader throughput: Mahout 6,101 vs PennyLane 1,604 vectors/s](./mnist-amplitude-benchmark.png)

All benchmarks support AMD backend selection via `--qdp-backend amd`, so you can do a direct CUDA-vs-ROCm comparison on the same workload.

## Documentation Overhaul

The docs site received a significant refresh:

- **Frontmatter added to every `docs/**/*.md` page** — fixes SEO metadata that was missing from most pages.
- **Self-hosted KaTeX** — math rendering now works offline and no longer depends on the CDN, fixing the "KaTeX not found" error reported by several contributors.
- **Troubleshooting guide** — a new guide covers the most common install and runtime issues, especially around GPU detection and CUDA/ROCm version mismatches.
- **CONTRIBUTING merged into the README** — reduces the number of places a new contributor needs to read to get started.
- **Type hints added to qumat** — initial pass adding Python type annotations, laying the groundwork for better IDE support and static analysis.
- **PR policy and review guidelines** — documents the merge criteria and review expectations in one place.

## Other Improvements

- **Cloud storage support** — the QDP data loader now supports S3 and GCS remote URLs in addition to local paths, enabling benchmarks and pipelines that read directly from object storage.
- **Encoding and Dtype enums** — `Encoding` and `Dtype` are now proper Python enums rather than bare strings, with static dispatch in the encoder.
- **Pure-PyTorch reference implementations** — added alongside the CUDA kernels for correctness comparison and CPU fallback.
- **Ruff rules expanded** — `E` and most `ANN` rules are now enforced; document type-checking CI added.
- **pytest-xdist** — parallel test execution cuts CI wall-clock time on multi-core machines.

## Getting Started

```bash
# Core package
pip install qumat==0.6.0

# With GPU-accelerated QDP extension (Linux x86_64 + NVIDIA CUDA)
pip install "qumat[qdp]==0.6.0"
```

- PyPI: https://pypi.org/project/qumat/0.6.0/
- Release tag: https://github.com/apache/mahout/releases/tag/mahout-qumat-0.6.0
- Docs: https://mahout.apache.org/

We welcome feedback and contributions on the [dev@mahout.apache.org](mailto:dev@mahout.apache.org) mailing list and on [GitHub](https://github.com/apache/mahout).
