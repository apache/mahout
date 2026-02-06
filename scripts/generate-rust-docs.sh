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

# Generate Rust API documentation using cargo doc
# Output is placed in website/static/api/rust/ for Docusaurus integration

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
QDP_DIR="$ROOT_DIR/qdp"
OUTPUT_DIR="$ROOT_DIR/website/static/api/rust"

echo "Generating Rust API documentation..."
echo "Output directory: $OUTPUT_DIR"

# Change to QDP workspace directory
cd "$QDP_DIR"

# Generate rustdoc for qdp-core package
# --no-deps: Don't document dependencies
# Using default features to avoid CUDA/GPU requirements
cargo doc --no-deps --package qdp-core

# Copy generated documentation to website static directory
mkdir -p "$OUTPUT_DIR"
cp -r "$QDP_DIR/target/doc/"* "$OUTPUT_DIR/"

echo "Rust API documentation generated successfully!"
echo "Entry point: $OUTPUT_DIR/qdp_core/index.html"
