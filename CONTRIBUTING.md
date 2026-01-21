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

# Contributing to Apache Mahout (Qumat)

Thank you for your interest in contributing to Apache Mahout!

## 1. Installation

**Prerequisites:** Python 3.10 (>=3.10,<3.14), uv, Git

### Install uv

```bash
pip install uv
```

or follow the instructions in the [uv documentation](https://docs.astral.sh/uv/).

### Clone and Install Dependencies

```bash
git clone https://github.com/apache/mahout.git
cd mahout
uv sync --group dev
```

### Set Up Pre-commit Hooks

```bash
pre-commit install
```

## 2. Development Workflow

### 2.1 Open an Issue

Create a new issue in [GitHub](https://github.com/apache/mahout/issues) and discuss your ideas with the community.

### 2.2 Make Changes

Create a new branch for your changes:

```bash
git checkout -b your-feature-name
```

Make your changes, then commit (pre-commit hooks will run automatically):

```bash
git add .
git commit -m "Description of your changes"
git push
```

### 2.3 Test

The project uses a unified test workflow with pytest. Tests are organized in the `testing/` directory.

**Test Structure:**
- `testing/qumat/` - Tests for the Qumat quantum computing library
- `testing/qdp/` - Tests for the Quantum Data Plane (GPU-accelerated, auto-skipped if extension unavailable)
- `testing/utils/` - Shared test utilities and helpers
- `testing/conftest.py` - Pytest configuration with shared fixtures

See [testing/README.md](testing/README.md) for detailed testing documentation.

### 2.4 Pre-commit Checks

Run pre-commit hooks:

```bash
pre-commit run
```

Or run pre-commit hooks on all files:

```bash
pre-commit run --all-files
```

### 2.5 Create a Pull Request

Create a pull request on GitHub. Please follow the [pull request template](.github/PULL_REQUEST_TEMPLATE) to provide a detailed description of your changes.

## 3. Website Development

The website is built with [Docusaurus](https://docusaurus.io/). The `/docs` directory is the source of truth for documentation.

### Local Development

```bash
cd website-new
npm install
npm run start
```

This starts a local development server at `http://localhost:3000` with hot reload.

### Building

```bash
cd website-new
npm run build
```

The build syncs documentation from `/docs` and `/qdp/docs` automatically.

### Documentation Workflow

1. Edit documentation in the `/docs` directory (not `website-new/docs`)
2. Run `npm run sync` in `website-new/` to update the website docs
3. The sync runs automatically during `npm run start` and `npm run build`

### Creating a Version Snapshot

When releasing a new version, snapshot the current documentation:

```bash
cd website-new
npm run docusaurus docs:version X.Y
```

## 4. Project Structure

- `qumat/` - Core library code
- `qdp/` - QDP (Quantum Data Plane)
- `docs/` - Documentation (source of truth)
- `examples/` - Examples and Jupyter notebooks
- `testing/` - Test files (using pytest)
- `website-new/` - Website source code (using Docusaurus)
