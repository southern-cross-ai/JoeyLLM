# 🦘✨ Contributing to JoeyLLM

Welcome to the JoeyLLM project! We’re building an Australian Large Language Model (LLM) using PyTorch, with a focus on efficiency and scalability. We truly believe that every contribution makes a difference—no matter how big or small. Whether you're fixing a typo, improving documentation, optimizing performance, or adding new features, your efforts help us build something remarkable.

We love to see new faces joining the community! Whether you're a seasoned developer or just starting out, there’s always a way to get involved. Don’t hesitate to reach out if you need help getting started—we’re here to support you. Your ideas and contributions are what make this project grow stronger and better!

## 📝 How to Get Started

Our workflow is centered around Docker containers to keep things consistent and easy to manage. By running everything inside Docker, we avoid conflicts between different environments and make it simple to test changes. Plus, using Docker lets us run large-scale tests on powerful GPU clusters (4 to 8 A100 GPUs) during nightly testing sessions, so we know our models are solid and optimized.

Whenever you’re working on the project, make sure to always pull the latest code into the container. This ensures that your tests are running against the most current version and helps us keep things organized. Everything is tested directly inside the Docker environment to make sure it behaves the same on all systems.

If you’ve got a compatible GPU on your machine, double-check that Docker and the GPU layer are properly set up before running tests. This way, you can take advantage of hardware acceleration when developing and experimenting with the model.

Let's dive into the steps to start contributing!

## 🛠️ Prerequisites

### Docker Installation:

Make sure you have Docker installed and running on your system.

Install the Docker Engine and GPU support if you have a GPU:

- [Docker Installation Guide](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit (for GPU support)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)

#### Docker GPU Test:

To ensure your GPU is accessible, run:

```bash
docker run --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

You should see your GPU listed if it’s configured correctly.

## 🚀 Workflow Overview

### Fork the Repository

1. Go to the JoeyLLM GitHub page and click the "Fork" button.
2. Clone your fork to your local machine:

```bash
git clone https://github.com/yourusername/JoeyLLM.git
cd JoeyLLM
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/Southern-Cross-AI/JoeyLLM.git
```

4. Verify your remotes:

```bash
git remote -v
```

### Build the Docker Container

We run all experiments and tests inside Docker to maintain consistency:

```bash
docker build -t joeyllm:latest .
```

This builds the container with the necessary dependencies.

### Start the Docker Container

Run the container with GPU support (if available):

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace joeyllm:latest /bin/bash
```

Inside the container, navigate to the workspace:

```bash
cd /workspace
```

### Make Your Code Changes

- Modify the model or any other components as needed.
- If system dependencies change, update the Dockerfile.

### Run Tests

Make sure to test your changes locally before submitting:

```bash
pytest tests/
```

For GPU-specific tests, run:

```bash
pytest --cuda tests/
```

Once testing is complete, exit the container:

```bash
exit
```

Make sure the container shuts down properly.

## 💡 Submitting Your Contribution

### Commit Your Changes

Stage and commit your changes:

```bash
git add .
git commit -m "Add feature: Enhanced model optimization"
```

### Push Your Branch

```bash
git push origin feature/my-new-feature
```

### Open a Pull Request (PR)

1. Go to your fork on GitHub and click "Compare & pull request".
2. Clearly explain your changes and why they’re needed.
3. Link to any related issues.

## ✅ Pull Request Review and Testing

Once your PR is submitted:

- The code will undergo nightly testing on 4 to 8 A100 GPUs.
- Maintainers will review your changes and may request updates.
- Once approved, your PR will be merged into the main branch.

## 🤝 Community and Support

Got stuck or need help?

- Join our community on [Discord](https://discord.gg/southerncrossai)
- Contact us via email at [support@southerncross.ai](mailto:support@southerncross.ai)

## 💡 Additional Tips

- Always make sure your Docker environment is set up correctly before running tests.
- Only test inside the Docker container to maintain consistency.
- Clearly document any system dependencies you add to the Dockerfile.

Thank you for contributing to JoeyLLM! Your support and efforts make a difference! 🎉
