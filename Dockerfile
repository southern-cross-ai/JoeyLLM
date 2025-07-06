# Extend from your main JoeyLLM image that already includes curl
FROM southerncrossai/joeyllm

# Set noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install bash-completion and VS Code Server
RUN apt-get update && \
    apt-get install -y bash-completion && \
    curl -fsSL https://code-server.dev/install.sh | sh && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add vscode user (no sudo)
RUN useradd -ms /bin/bash vscode

# Create workspace directory and set permissions
RUN mkdir -p /home/vscode/workspace && chown vscode:vscode /home/vscode/workspace

# Switch to vscode user
USER vscode
WORKDIR /home/vscode

# Expose the VS Code Server port
EXPOSE 8080

# Set default entry point and command
ENTRYPOINT ["code-server"]
CMD ["--bind-addr", "0.0.0.0:8080", "/home/vscode/workspace"]
