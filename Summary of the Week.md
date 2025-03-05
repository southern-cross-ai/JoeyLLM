Setting up the Environment:

To ensure a consistent working environment where all programs are properly configured, we set up Docker with VS Code using Ubuntu 22.04 as the base system. First, we installed VS Code along with the necessary extensions: Remote - Containers and Docker (both by Microsoft). Then, we installed Docker Desktop (Windows/macOS) or Docker Engine (Linux) and ensured it was running with docker ps. To maintain consistency, we used Ubuntu 22.04 as the container OS and started a new container with docker run -dit --name my_container ubuntu:22.04 /bin/bash. If a container with the same name existed, we either renamed it (docker rename my_container my_old_container), removed it (docker stop my_container && docker rm my_container), or created a new one with a different name. To connect VS Code to the running container, we opened VS Code, pressed Ctrl + Shift + P, searched for "Remote-Containers: Attach to Running Container", and selected the container. If no results appeared, we ensured the Remote - Containers extension was installed. To verify running containers, we used docker ps, stopped a container with docker stop my_container, and removed it with docker rm my_container. If we needed to work inside a running container, we used docker exec -it my_container bash. Additionally, to maintain a structured development environment inside a container, we created a .devcontainer folder with a devcontainer.json file, defining the necessary settings, and reopened the project in the container using Ctrl + Shift + P → "Remote-Containers: Reopen in Container". This setup ensures that all programs are properly configured and work consistently across different machines.


# Project Plan for Next Week

## 1. Client Meeting to Discuss Progress
	Time: 2pm Wednesday, 12 March
	Objectives:
	•	Report this week’s progress, including data collection, model training preparation, and technical implementation
	•	Discuss current challenges and seek client feedback
	•	Define the next phase’s objectives and priorities

## 2. Build Project Documentation & Track Work Progress
	•	Update and organize project documentation, including:
	•	Requirement analysis and technical plan
	•	Data processing and model training workflow
	•	Completed tasks and ongoing work tracking

## 3. Collect Corpus Data for LLM Training
	•	Identify and gather relevant textual data
	•	Preprocess and clean the collected data to ensure quality
	•	Organize the dataset for integration into the training pipeline
