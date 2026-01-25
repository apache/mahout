# Detect NVIDIA GPU
HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo yes || echo no)

test_rust:
ifeq ($(HAS_NVIDIA),yes)
	cd qdp && cargo test
else
	@echo "No NVIDIA GPU detected, skipping test_rust"
endif

test_python:
	uv sync --group dev
ifeq ($(HAS_NVIDIA),yes)
	unset CONDA_PREFIX && uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
else
	@echo "No NVIDIA GPU detected, skipping maturin develop"
endif
	uv run pytest

test: test_rust test_python

pre-commit:
	uv sync --group dev
	uv run pre-commit run --all-files
