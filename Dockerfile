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
    && rm -rf /var/lib/apt/lists/*  

# Ensure 'python3' is recognised as 'python'
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Default command (for interactive HPC usage)
CMD ["/bin/bash"]
