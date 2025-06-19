# Use Ubuntu 24.04 for stability
FROM ubuntu:24.04

# Set environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    bash-completion \
    vim \
    git \
    wget \
    curl \
    htop \
    nvtop \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    openssh-client \
    git-lfs \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*  

# Ensure 'python3' is recognised as 'python'
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ------------------------------
# Install CUDA Toolkit (12.6)
# ------------------------------
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
    | gpg --dearmor | tee /etc/apt/keyrings/nvidia.gpg > /dev/null

RUN echo "deb [signed-by=/etc/apt/keyrings/nvidia.gpg] \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
    > /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
    apt-get install -y cuda-toolkit-12-6 && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH=/usr/local/cuda-12.6/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}

# Default command (for interactive HPC usage)
CMD ["/bin/bash"]
