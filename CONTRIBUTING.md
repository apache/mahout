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

Run the test suite using pytest:

```bash
pytest
```

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

## 3. Project Structure

- `qumat/` - Core library code
- `qdp/` - QDP (Quantum Data Plane)
- `docs/` - Documentation
- `examples/` - Examples and Jupyter notebooks
- `testing/` - Test files (using pytest)
- `website/` - Website source code (using Jekyll)
