# Getting Started with Qumat

## Basic Installation

Getting started with Qumat is easy, thanks to the simplified installation process. You can install Qumat by choosing one of the following methods.

### Method 1: Install from PyPI (Recommended)

```bash
pip install qumat
```

### Method 2: Install from Source (Development)

For development or to get the latest changes, use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/apache/mahout
cd mahout
pip install uv
uv sync                     # Core Qumat
uv sync --extra qdp         # With QDP (requires NVIDIA GPU + CUDA)
```

:::note Why uv?
The project uses `uv` to handle dependency overrides required for Python 3.10+ compatibility with some backend dependencies.
:::

## Dependencies

Prior to installation, ensure Python 3.10-3.12 is installed. Dependencies such as Qiskit, Cirq, and Amazon Braket SDK will be managed automatically.

## Examples

Refer to the example notebooks in the `examples/` directory at the repository root for practical implementations:

- `examples/Simple_Example.ipynb` - Basic quantum circuit example
- `examples/Optimization_Example.ipynb` - Optimization with parameterized circuits

## Building the Website

To serve the website locally:

```bash
cd website
npm install
npm run start
```

See the [website README](https://github.com/apache/mahout/tree/main/website#readme) for more details.
