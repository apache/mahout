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

.PHONY: test_rust test_python tests pre-commit setup-test-python

# Detect NVIDIA GPU
HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo yes || echo no)

setup-test-python:
	uv sync --group dev

test_rust:
ifeq ($(HAS_NVIDIA),yes)
	cd qdp && cargo test -p qdp-core -p qdp-kernels
else
	@echo "[SKIP] No NVIDIA GPU detected, skipping test_rust"
endif

test_python: setup-test-python
ifeq ($(HAS_NVIDIA),yes)
	unset CONDA_PREFIX && uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml
else
	@echo "[SKIP] No NVIDIA GPU detected, skipping maturin develop"
endif
	uv run pytest

tests: test_rust test_python

pre-commit: setup-test-python
	uv run pre-commit run --all-files
