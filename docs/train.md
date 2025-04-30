# GPT-2 Training Guide

This document explains how to train the custom GPT-2 model.

## Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.4 +
Pytorch with gpu support
- Required packages:
  ```bash
  pip install -r requirements.txt

## 📁 Project Structure Overview
JoeyLLM/
├── src/
│   ├── configs/config.yaml      # YAML-based model & training configuration
│   ├── data/
│   │   ├── data.py              # Loads and batches the training dataset
│   │   └── chunk.py             # Preprocessing script to chunk tokenized data
│   └── model/model.py           # Custom GPT-2 model and transformer blocks
├── src/main.py                  # Main training script using Hydra config
├── requirements.txt             # Python dependencies


## Monitor with Weights & Biases

-Before running training, login to Weights & Biases: `wandb login`

## Training on Single GPU
    python src/main.py
  

This will:

Load the custom GPT-2 model with chosen configuration

Load 25% of the Project_Gutenberg_Australia dataset

Begin training with full logging to Weights & Biases

Save checkpoints after every epoch
