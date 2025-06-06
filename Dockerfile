# Start FROM your main runtime image
FROM your-main-image

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Optionally expose the code-server port
EXPOSE 8443

# Default command to run code-server (optional)
CMD ["code-server", "--bind-addr", "0.0.0.0:8443", "/workspace"]

