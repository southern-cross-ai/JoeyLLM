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
├── src/config/config.py    # Contains model and training configuration classes  
├── src/data/data.py    # Loads and prepares the dataset  
├── src/model/model.py    # Custom GPT-2 architecture implementation  
├── src/main.py    # Main training script (currently works with for single GPU)
├── requirements.txt       # Dependencies  

## Monitor with Weights & Biases

-Before running training, login to Weights & Biases: `wandb login`

## Training on Single GPU
    python src/main.py
  

This will:

Load the custom GPT-2 model with chosen configuration

Load 25% of the Project_Gutenberg_Australia dataset

Begin training with full logging to Weights & Biases

Save checkpoints after every epoch