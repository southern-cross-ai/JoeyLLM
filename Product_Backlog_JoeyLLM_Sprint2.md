
# 🗃️ Product Backlog – JoeyLLM Project (Based on Jira Kanban)

_Organized as Product Backlog Items (PBIs) from the current Jira board across all columns_

---

## 🔰 EPIC: Project Setup & Environment

### ✅ PBI-001: JoeyLLM Project and Team Introduction
**User Story**: As a team, we need to understand the project scope and form a working group so we can start development effectively.  
**Acceptance Criteria**:
- Team introduced and communication channels set up
- Project goals discussed with stakeholders

---

### ✅ PBI-002: Setting up Public and Private SSH Keys
**User Story**: As a developer, I want to configure SSH keys so I can access repositories securely.  
**Acceptance Criteria**:
- SSH keys generated and added to GitHub
- Access tested and confirmed by team

---

### ✅ PBI-003: Setting up Python Virtual Environment
**User Story**: As a developer, I want an isolated Python environment so dependencies don’t conflict.  
**Acceptance Criteria**:
- Virtual environment created and activated
- `requirements.txt` installed successfully

---

### ✅ PBI-004: Setting up Docker Environment
**User Story**: As a team, we want a portable environment so development is reproducible across systems.  
**Acceptance Criteria**:
- Dockerfile created and working container built
- Readme includes build/run instructions

---

### ✅ PBI-005: Setting up Communication Channels
**User Story**: As a team, we need clear channels like Slack/Discord for regular collaboration.  
**Acceptance Criteria**:
- Communication platform chosen and active
- Guidelines shared for effective use

---

## 📦 EPIC: Data Collection & Processing

### 🟨 PBI-006: Data Cleaning (KAN-11)
**User Story**: As a data engineer, I want to clean our input data so the model learns from high-quality content.  
**Acceptance Criteria**:
- Noise, HTML, duplicates removed
- Logging implemented for cleaned samples

---

### 🟨 PBI-007: Learn FineWeb Dataset Cleaning Methods (KAN-21)
**User Story**: As a developer, I want to study FineWeb's data processing steps to improve our own pipeline.  
**Acceptance Criteria**:
- Notes compiled on methods used
- Report or implementation draft prepared

---

### 🟨 PBI-008: Learn GPT-2 Pipeline via Tutorials (KAN-22)
**User Story**: As a team, we need to learn the fine-tuning and training pipeline so we can build our custom GPT-2.  
**Acceptance Criteria**:
- Team members complete 2–3 HuggingFace tutorials
- Summary notes uploaded

---

### 🟧 PBI-009: Tokenize Cleaned Data (KAN-30)
**User Story**: As a dev, I want to tokenize the cleaned dataset so it’s ready for training.  
**Acceptance Criteria**:
- Tokenization script created using GPT2Tokenizer
- Output saved in batch format

---

### ✅ PBI-010: Data Skimming (KAN-16)
**User Story**: As a researcher, I want to quickly skim and understand the dataset so we can plan cleaning strategies.  
**Acceptance Criteria**:
- Exploratory analysis completed
- Skim results shared with team

---

## 🤖 EPIC: Model Setup, Training, and Experimentation

### 🟧 PBI-011: Experiment with GPT and WandB (KAN-10)
**User Story**: As a researcher, I want to experiment with GPT-2 and log results using WandB.  
**Acceptance Criteria**:
- Pretrained GPT-2 tested
- Logs pushed to WandB
- Fine-tuning initialized on local and A100 setups

---

### 🟧 PBI-012: Work on FineWeb with HuggingFace (KAN-9)
**User Story**: As a dev, I want to work with the FineWeb dataset using HuggingFace datasets API to integrate our training flow.  
**Acceptance Criteria**:
- Dataset loaded, subset sampled
- Documented pipeline for cleaning/tokenizing

---

### 🟧 PBI-013: Upload GPT-2 Theory Material (KAN-27)
**User Story**: As a team, we want to upload learning materials for GPT-2 so everyone understands the architecture.  
**Acceptance Criteria**:
- Notes and slides added to shared repo
- Content peer-reviewed

---

### ✅ PBI-014: Summary of GPT-2 Build by Karpathy (KAN-25)
**User Story**: As a learner, I want to summarize Karpathy’s GPT-2 videos so I understand the core concepts.  
**Acceptance Criteria**:
- Summary notes uploaded
- Key concepts explained

---

## 📋 EPIC: Communication & Collaboration

### ✅ PBI-015: Client and Team Standups (KAN-19)
**User Story**: As a team, we want to conduct regular check-ins to stay aligned and resolve blockers.  
**Acceptance Criteria**:
- Standups conducted weekly
- Minutes or summaries logged

---

### ✅ PBI-016: Take Minutes of Tutorials 1 & 2 (KAN-31)
### ✅ PBI-017: Take Minutes of Tutorial 3 (KAN-32)
**User Story**: As a team, we want to document all tutorials and feedback to refer later.  
**Acceptance Criteria**:
- Meeting minutes uploaded and shared
- Reflections summarized

---

### ✅ PBI-018: Upload Code Reviews (KAN-28)
**User Story**: As a team, we want to track and review each other’s code for better quality and consistency.  
**Acceptance Criteria**:
- Code review process defined
- Logs/screenshots uploaded

---

### ✅ PBI-019: Creating Centralised Data Repository (KAN-8)
**User Story**: As a dev team, we want a single shared space to manage our datasets.  
**Acceptance Criteria**:
- Repo/folder created and structured
- Access granted to all members

---

### ✅ PBI-020: Explainer for Setting up JoeyLLM on Mac (KAN-20)
**User Story**: As a developer, I want a step-by-step guide to run JoeyLLM on MacOS.  
**Acceptance Criteria**:
- Explainer created and tested
- Peer feedback received

---

📝 Total PBIs: 20  
🟩 Done: 11  
🟧 In Progress: 4  
🟨 Sprint Backlog: 3  
