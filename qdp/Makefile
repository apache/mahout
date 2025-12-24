.PHONY: install build build_nvtx_profile run_nvtx_profile test test_python test_rust e2e_test clean help

help:
	@echo "Available targets:"
	@echo "  make install             - Install the mahout python package"
	@echo "  make build               - Build qdp-core"
	@echo "  make build_nvtx_profile  - Build qdp-core with observability features"
	@echo "  make run_nvtx_profile    - Build example, run with nsys, and show stats (EXAMPLE=nvtx_profile)"
	@echo "  make test                - Run all unit tests (Python + Rust)"
	@echo "  make test_python         - Run Python unit tests only"
	@echo "  make test_rust           - Run Rust unit tests only"
	@echo "  make e2e_test            - Run all e2e tests under qdp-python/benchmark"
	@echo "  make clean               - Clean build artifacts"

install:
	@echo "Installing mahout python package..."
	cd qdp-python && uv sync --group dev
	cd qdp-python && uv run maturin develop

build:
	@echo "Building qdp-core..."
	cargo build -p qdp-core

build_nvtx_profile:
	@echo "Building qdp-core with observability features..."
	cargo build -p qdp-core --features observability --release

test: test_python test_rust

test_python:
	@echo "Running Python unit tests..."
	cd qdp-python && uv run pytest tests/

test_rust:
	@echo "Running Rust unit tests..."
	cargo test --workspace

e2e_test:
	@echo "Running e2e benchmark tests..."
	cd qdp-python/benchmark && uv run python benchmark_e2e.py
	cd qdp-python/benchmark && uv run python benchmark_dataloader_throughput.py

run_nvtx_profile:
	$(eval EXAMPLE ?= nvtx_profile)
	@echo "Building example '$(EXAMPLE)' with observability features..."
	cargo build -p qdp-core --example $(EXAMPLE) --features observability --release
	@echo "Running '$(EXAMPLE)' with nsys profiling..."
	nsys profile --trace=cuda,nvtx --force-overwrite=true -o report ./target/release/examples/$(EXAMPLE)
	@echo "Showing profiling statistics..."
	nsys stats --force-export=true report.nsys-rep

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	cd qdp-python && rm -rf target/
	cd qdp-python && rm -rf .pytest_cache/
	cd qdp-python && rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
