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
├── main.py                     # Entry point script: loads config and starts training
├── requirements.txt            # Project dependencies
└── src/
    ├── configs/
    │   └── config.yaml         # YAML configuration (Hydra) for model, training, logging

    ├── data/
    │   ├── dataset.py          # Loads and batches tokenized training data
    │   ├── test_data.py        # Loads/handles validation or test datasets
    │   └── chunk.py            # Preprocessing script to split long sequences into chunks

    ├── model/
    │   ├── joeyllm.py          # Custom GPT-2 model (transformers, decoder blocks, attention)
    │   └── test_model.py       # Unit tests or evaluation scripts for model components

    ├── tokenizer/
    │   ├── train_tokenizer.py  # Trains a tokenizer using a raw text corpus
    │   └── test_tokenizer.py   # Validates tokenizer output and decoding accuracy

    └── train/
        ├── loop.py             # Core training loop (epochs, logging, checkpointing)
        └── optimizer.py        # Optimizer setup (AdamW)


## Monitor with Weights & Biases

-Before running training, login to Weights & Biases: `wandb login`

## Training on Single GPU
    python src/main.py
  

This will:

Load the custom GPT-2 model with chosen configuration

Load 25% of the Project_Gutenberg_Australia dataset

Begin training with full logging to Weights & Biases

Save checkpoints after every epoch
