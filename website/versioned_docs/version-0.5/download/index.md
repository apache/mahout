---
title: Install
sidebar_label: Install
---

# Install

```bash
pip install qumat
```

To install with QDP (Quantum Data Plane) support:

```bash
pip install qumat[qdp]
```

## From Source

```bash
git clone https://github.com/apache/mahout.git
cd mahout
pip install uv
uv sync                     # Core Qumat
uv sync --extra qdp         # With QDP (requires CUDA GPU)
```

## Apache Release

Official source releases are available at [apache.org/dist/mahout](http://www.apache.org/dist/mahout).

To verify the integrity of a downloaded release:

```bash
gpg --import KEYS
gpg --verify mahout-qumat-0.5.zip.asc mahout-qumat-0.5.zip
```

## Links

- [PyPI](https://pypi.org/project/qumat/)
- [Apache SVN](http://www.apache.org/dist/mahout)
