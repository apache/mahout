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

# peotry
apt update
apt install apt-utils -y
apt install pipx -y
pipx ensurepath
pipx install poetry

# setup pre-install hook
poetry install --extras dev
poetry run pre-commit install

# (Optional) CUDA dev packages you may want later
# apt-get install -y cuda-toolkit-12-4
