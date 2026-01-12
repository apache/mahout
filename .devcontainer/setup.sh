#!/usr/bin/env bash
set -eux

# Install Rust + Cargo
if ! command -v cargo >/dev/null 2>&1; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    echo 'source $HOME/.cargo/env' >> ~/.bashrc
fi

# Common dev tools
apt-get update
apt-get install -y \
    build-essential \
    pkg-config \
    git \
    vim


# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# setup pre-commit hooks
cd /workspaces/mahout
uv sync --group dev
uv run pre-commit install
