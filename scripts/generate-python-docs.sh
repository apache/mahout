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

# Generate Python API documentation as Markdown using pydoc-markdown
# Output is placed in docs/api/python/ for Docusaurus integration

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$ROOT_DIR/docs/api/python"

echo "Generating Python API documentation..."
echo "Output directory: $OUTPUT_DIR"

# Ensure pydoc-markdown is available
if ! command -v pydoc-markdown &>/dev/null; then
  echo "Error: pydoc-markdown not found. Install with: pip install pydoc-markdown" >&2
  exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Change to root directory for proper module resolution
cd "$ROOT_DIR"

# Generate markdown for qumat module
pydoc-markdown -I qumat -m qumat > "$OUTPUT_DIR/index.md"
