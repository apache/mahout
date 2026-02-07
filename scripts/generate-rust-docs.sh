#!/usr/bin/env bash
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
#

# Generate Rust API documentation as Markdown using rustdoc-md
# Output is placed in docs/api/rust/ for Docusaurus integration

set -euo pipefail

# Add ~/.cargo/bin and rustup toolchain to PATH if not already there
[[ -d "$HOME/.cargo/bin" ]] && export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo &>/dev/null; then
  RUSTUP_CARGO="$(find "$HOME/.rustup/toolchains" -path '*/bin/cargo' -print -quit 2>/dev/null || true)"
  if [[ -n "$RUSTUP_CARGO" ]]; then
    export PATH="$(dirname "$RUSTUP_CARGO"):$PATH"
  else
    echo "Error: cargo not found. Please install Rust: https://rustup.rs" >&2
    exit 1
  fi
fi

# Ensure rustdoc-md is available
if ! command -v rustdoc-md &>/dev/null; then
  echo "Error: rustdoc-md not found. Install with: cargo install rustdoc-md" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
QDP_DIR="$ROOT_DIR/qdp"
OUTPUT_DIR="$ROOT_DIR/docs/api/rust"

echo "Generating Rust API documentation..."
echo "Output directory: $OUTPUT_DIR"

# Change to QDP workspace directory
cd "$QDP_DIR"

# Find nightly cargo (cargo +nightly requires rustup in PATH, so use direct path)
NIGHTLY_CARGO="$(find "$HOME/.rustup/toolchains/nightly"* -path '*/bin/cargo' -print -quit 2>/dev/null || true)"
if [[ -z "$NIGHTLY_CARGO" ]]; then
  echo "Error: nightly toolchain not found. Install with: rustup install nightly" >&2
  exit 1
fi

# Step 1: Generate rustdoc JSON (requires nightly)
# Prepend nightly bin to PATH so both cargo AND rustdoc resolve from nightly
NIGHTLY_BIN="$(dirname "$NIGHTLY_CARGO")"
echo "Using nightly toolchain at: $NIGHTLY_BIN"
echo "Generating rustdoc JSON..."
PATH="$NIGHTLY_BIN:$PATH" RUSTDOCFLAGS="-Z unstable-options --output-format json" \
  cargo doc --no-deps --package qdp-core

# Step 2: Convert JSON to Markdown
JSON_FILE="$QDP_DIR/target/doc/qdp_core.json"
mkdir -p "$OUTPUT_DIR"
rustdoc-md --path "$JSON_FILE" --output "$OUTPUT_DIR/index.md"

# Step 3: Add Docusaurus frontmatter
TMPFILE=$(mktemp)
cat > "$TMPFILE" <<'FRONTMATTER'
---
title: Rust API Reference
sidebar_label: Rust API (qdp-core)
sidebar_position: 2
---

FRONTMATTER
cat "$OUTPUT_DIR/index.md" >> "$TMPFILE"
mv "$TMPFILE" "$OUTPUT_DIR/index.md"

echo "Rust API documentation generated successfully!"
echo "Output: $OUTPUT_DIR/index.md"
