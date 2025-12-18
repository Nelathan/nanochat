FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/venv/bin:${PATH}"
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH="/workspace"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/workspace/hf-cache

# Install system dependencies with cleanup
RUN apt-get update --yes && \
  apt-get install --yes --no-install-recommends \
  git wget curl nano unzip \
  build-essential pkg-config \
  gcc g++ cmake make && \
  apt-get autoremove -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install Rust and add to PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
  echo 'source $HOME/.cargo/env' >> ~/.bashrc

# Install and configure package managers
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["bash"]
