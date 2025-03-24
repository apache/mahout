# Getting Started with Qumat

## Basic Installation

Getting started with Qumat is easy, thanks to the simplified installation process. You can install Qumat by choosing one of the following methods.

### Method 1: Clone and Install Locally

```
git clone https://github.com/apache/mahout
cd mahout
pip install .
```

### Method 2: Install via Git directly

```
pip install git+https://github.com/apache/mahout
```

Users might think these instructions are "too easy" due to the previous complexity associated with installing Mahout. Rest assured, as part of the Qumat reboot, we have made significant strides in simplifying both installation and getting started.

## Dependencies

Prior to installation, ensure Python 3.10+ is installed. Dependencies such as Qiskit, Cirq, and Amazon Braket SDK will be managed by pip.

## Installation from Source

If you wish to build from source, use the following command:

```
git clone https://github.com/apache/mahout
cd mahout
pip install .
```

## Examples

Refer to the example notebooks in the 'examples' directory for practical implementations of basic commands and getting started. Notable examples include: 

- [Simple_Example.ipynb](examples/Simple_Example.ipynb)
- [Optimization_Example.ipynb](examples/Optimization_Example.ipynb)

## Building the Website

To serve the website locally, navigate to the 'website' directory and execute the following commands:

```
cd website
bundle exec jekyll serve
```

The "build_site.sh" script is for deployment tooling and not necessary for users wanting to serve the website locally.