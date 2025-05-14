FROM ubuntu:22.04

# Set environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the system and install the necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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
    && rm -rf /var/lib/apt/lists/*  

# Ensure 'python3' is recognised as 'python'
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy local code (from build context) into image
COPY . /workspace

# Default command (for interactive HPC usage)
CMD ["/bin/bash"]
