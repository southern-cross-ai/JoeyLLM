# JoeyLLM GPT-2 model Training Guide

This document explains how to train the custom JoeyLLM GPT-2 model.

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
│   ├── configs/
│   │   └── config.yaml           # Hydra-compatible YAML config for model, training, logging  
│
│   ├── data/
│   │   ├── dataset.py            # Loads and batches tokenized dataset for training  
│   │   ├── test_data.py          # Handles test/validation datasets  
│   │   └── chunk.py              # Preprocesses raw token sequences into fixed-length chunks  
│
│   ├── model/
│   │   ├── joeyllm.py            # Core GPT-2 model architecture (decoder blocks, attention, etc.)  
│   │   └── test_model.py         # Unit tests or experimental evaluation of the model  
│
│   ├── tokenizer/
│   │   ├── train_tokenizer.py    # Trains a custom tokenizer from raw text data  
│   │   └── test_tokenizer.py     # Validates tokenizer behavior (encoding/decoding tests)  
│
│   ├── train/
│   │   ├── loop.py               # Training loop logic (epochs, logging, loss, etc.)  
│   │   └── optimizer.py          # Optimizer and scheduler setup (e.g. AdamW)  
│
│   └── main.py                   # Entry point script: sets up config, model, data, and runs training  
│
├── requirements.txt              # Python dependencies for the full pipeline  



## Monitor with Weights & Biases

-Before running training, login to Weights & Biases: `wandb login`

## Training on Single GPU
    python src/main.py
  

This will:

Load the custom GPT-2 model with chosen configuration

Load 25% of the Project_Gutenberg_Australia dataset

Begin training with full logging to Weights & Biases

Save checkpoints after every epoch
