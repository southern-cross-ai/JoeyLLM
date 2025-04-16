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
├── config.py              # Contains model and training configuration classes  
├── data.py                # Loads and prepares the dataset  
├── model.py               # Custom GPT-2 architecture implementation  
├── train_single_gpu.py    # Main training script for single GPU  
├── requirements.txt       # Dependencies  

## Monitor with Weights & Biases

-Before running training, login to Weights & Biases: `wandb login`

## Training on Single GPU
    python train_single_gpu.py
  

This will:

Load the custom GPT-2 model with chosen configuration

Load 25% of the Project_Gutenberg_Australia dataset

Begin training with full logging to Weights & Biases

Save checkpoints after every epoch