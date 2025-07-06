# Use Ubuntu 24.04 for stability
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Set environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the system and install the necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    bash-completion \
    vim \
    git \
    lsof \
    wget \
    curl \
    htop \
    nvtop \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    openssh-client \
    ca-certificates \
    libnvidia-ml-dev \
    git-lfs \
    sshfs \
    && rm -rf /var/lib/apt/lists/*

# Ensure 'python3' is recognised as 'python'
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install UV (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add ~/.local/bin to PATH for UV if needed
ENV PATH="${PATH}:/root/.local/bin"

# Default command (for interactive HPC usage)
CMD ["/bin/bash"]
