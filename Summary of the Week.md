# Setting Up Docker with VS Code for a Consistent Development Environment

## Overview
To ensure a **consistent working environment** where all programs are properly configured, we set up **Docker with VS Code** using **Ubuntu 22.04** as the base system. This ensures that development workflows remain stable across different machines.

## 1. Install Required Tools
### Install Visual Studio Code
- Download and install **[VS Code](https://code.visualstudio.com/download)**.

### Install Required VS Code Extensions
1. Open **VS Code**.
2. Press `Ctrl + Shift + X` to open the **Extensions** panel.
3. Search and install the following extensions:
   - **Remote - Containers** (by Microsoft)
   - **Docker** (by Microsoft)
4. Restart VS Code after installation.

### Install and Set Up Docker
#### Install Docker
- **Windows/macOS**: Install **[Docker Desktop](https://www.docker.com/get-started/)**.
- **Linux**: Install **Docker Engine** using the package manager:
  ```sh
  sudo apt install docker.io  # Ubuntu
  sudo dnf install docker-ce  # Fedora
  ```

- Ensure the Docker daemon is running:
  ```sh
  docker ps
  ```
  If Docker is not running, start it:
  ```sh
  systemctl start docker  # Linux
  ```

## 2. Running a Docker Container
### Start a New Container
To create and start a new container with Ubuntu 22.04:
```sh
docker run -dit --name my_container ubuntu:22.04 /bin/bash
```
- `-d`: Run container in detached mode.
- `-i`: Keep container input open.
- `-t`: Allocate a pseudo-terminal.
- `--name my_container`: Assigns a name to the container.

### Verify Running Containers
To check if the container is running:
```sh
docker ps
```
To list all containers (including stopped ones):
```sh
docker ps -a
```

## 3. Connecting VS Code to Docker
### Attach VS Code to a Running Container
1. Open **VS Code**.
2. Press `Ctrl + Shift + P` to open the **Command Palette**.
3. Search for:
   ```
   Remote-Containers: Attach to Running Container
   ```
4. Select your running container (e.g., `my_container`).
5. VS Code will now open the container’s file system.

## 4. Troubleshooting Docker Issues
### "Container Name Already in Use" Error
If you try to create a container with an existing name, Docker will throw an error:
```
docker: Error response from daemon: Conflict. The container name "/my_container" is already in use.
```
#### Solution 1: Use a Different Name
```sh
docker run -dit --name my_container2 ubuntu:22.04 /bin/bash
```
#### Solution 2: Remove the Existing Container
1. Stop the existing container:
   ```sh
   docker stop my_container
   ```
2. Remove the container:
   ```sh
   docker rm my_container
   ```
#### Solution 3: Rename the Existing Container
```sh
docker rename my_container my_old_container
```

# Project Role Sign-Up Sheet

| Role | Name |
|------|------|
| **Data Collection Specialist** |  |
| **Data Cleaning & Processing Engineer** |  |
| **Model Development Engineer** |  |
| **Web Development Engineer** |  |
| **Multimedia & Content Specialist** | Shuang Liu | |
| **Public Relations & Communication** | Shuang Liu | |

# Project Plan for Next Week

## 1. Client Meeting to Discuss Progress
	Time: 2pm Wednesday, 12 March
	Objectives:
	•	Report this week’s progress, including data collection, model training preparation, and technical implementation
	•	Discuss current challenges and seek client feedback
	•	Define the next phase’s objectives and priorities

## 2. Build Project Documentation & Track Work Progress
	Update and organize project documentation, including:
	•	Requirement analysis and technical plan
	•	Data processing and model training workflow
	•	Completed tasks and ongoing work tracking

## 3. Collect Corpus Data for LLM Training
	•	Identify and gather relevant textual data
	•	Preprocess and clean the collected data to ensure quality
	•	Organize the dataset for integration into the training pipeline

## 🎯Challenges and Solutions

| **Challenge**                                    | **Solution**                                          |
|--------------------------------------------------|------------------------------------------------------|
| **Challenge 1**: Difficulty in setting up Docker Environment among some MAC users.   |  Expert team members assisted them in setting up the environment.    |
| **Challenge 2**: Trouble understanding the difference between Containers and VM's | Matthew demonstrated the differences and advantages of Containers over VM's         |
| **Challenge 3**: Team structure undefined.     | Assigned roles like spokesperson, deputy, etc. to each team member.         |
| **Challenge 4**: Communication gap.       | Matthew created a Discord for one-stop communication.              |

