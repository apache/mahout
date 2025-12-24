.PHONY: install unit_test unit_test_python unit_test_rust e2e_test clean help

help:
	@echo "Available targets:"
	@echo "  make install          - Install the mahout python package"
	@echo "  make unit_test        - Run all unit tests (Python + Rust)"
	@echo "  make unit_test_python - Run Python unit tests only"
	@echo "  make unit_test_rust   - Run Rust unit tests only"
	@echo "  make e2e_test         - Run all e2e tests under qdp-python/benchmark"
	@echo "  make clean            - Clean build artifacts"

install:
	@echo "Installing mahout python package..."
	cd qdp-python && uv sync --group dev
	cd qdp-python && uv run maturin develop

unit_test: unit_test_python unit_test_rust

unit_test_python:
	@echo "Running Python unit tests..."
	cd qdp-python && uv run pytest tests/

unit_test_rust:
	@echo "Running Rust unit tests..."
	cargo test --workspace

e2e_test:
	@echo "Running e2e benchmark tests..."
	cd qdp-python/benchmark && uv run python benchmark_e2e.py
	cd qdp-python/benchmark && uv run python benchmark_dataloader_throughput.py

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	cd qdp-python && rm -rf target/
	cd qdp-python && rm -rf .pytest_cache/
	cd qdp-python && rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
