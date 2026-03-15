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

.PHONY: test_rust test_python tests pre-commit setup-test-python install-llvm-cov benchmark setup-benchmark

# Detect NVIDIA GPU
HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo yes || echo no)

setup-test-python:
	uv sync --group dev

install-llvm-cov:
	@cargo llvm-cov --version >/dev/null 2>&1 || (echo "[INFO] Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov)

test_rust: install-llvm-cov
ifeq ($(HAS_NVIDIA),yes)
	cd qdp && cargo llvm-cov test --workspace --exclude qdp-python --html --output-dir target/llvm-cov/html
	cd qdp && cargo llvm-cov report --summary-only
else
	@echo "[SKIP] No NVIDIA GPU detected, skipping test_rust"
endif

test_python: setup-test-python
ifeq ($(HAS_NVIDIA),yes)
	unset CONDA_PREFIX && uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
else
	@echo "[SKIP] No NVIDIA GPU detected, skipping maturin develop"
endif
	uv run pytest --cov=qumat --cov=qumat_qdp --cov-report=term-missing --cov-report=html:htmlcov

tests: test_rust test_python

pre-commit: setup-test-python
	uv run pre-commit run --all-files

setup-benchmark: setup-test-python
ifeq ($(HAS_NVIDIA),yes)
	@echo "[INFO] Setting up benchmark environment..."
	uv sync --group dev --extra qdp
	uv sync --project qdp/qdp-python --group benchmark --active
	unset CONDA_PREFIX && uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
else
	@echo "[SKIP] No NVIDIA GPU detected, skipping maturin develop"
	@echo "[INFO] Setting up benchmark environment (CPU-only)..."
	uv sync --project qdp/qdp-python --group benchmark --active
endif

benchmark: setup-benchmark
ifeq ($(HAS_NVIDIA),yes)
	@echo "[INFO] Running benchmarks..."
	@echo "[INFO] Available benchmark scripts:"
	@echo "  - benchmark_e2e.py: End-to-end latency (Disk -> GPU VRAM)"
	@echo "  - benchmark_latency.py: Data-to-State latency (CPU RAM -> GPU VRAM)"
	@echo "  - benchmark_throughput.py: DataLoader-style throughput"
	@echo ""
	@echo "[INFO] Run specific benchmarks with:"
	@echo "  uv run --active python qdp/qdp-python/benchmark/benchmark_e2e.py"
	@echo "  uv run --active python qdp/qdp-python/benchmark/benchmark_latency.py"
	@echo "  uv run --active python qdp/qdp-python/benchmark/benchmark_throughput.py"
	@echo ""
	@echo "[INFO] See qdp/qdp-python/benchmark/README.md for more options."
else
	@echo "[SKIP] No NVIDIA GPU detected, skipping benchmarks"
	@echo "[INFO] Benchmarks require NVIDIA GPU. Setup completed for manual execution."
endif
