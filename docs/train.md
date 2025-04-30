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
├── requirements.txt                # Python dependencies
└── src/
    ├── main.py                     # Entry point: loads config and starts training

    ├── configs/
    │   └── config.yaml             # Hydra-compatible configuration file

    ├── data/
    │   ├── dataset.py              # Loads and batches the training dataset
    │   ├── test_data.py            # Handles test/validation dataset loading
    │   └── chunk.py                # Chunks long tokenized text into fixed-size segments

    ├── model/
    │   ├── joeyllm.py              # GPT-2 model implementation (attention, transformer blocks)
    │   └── test_model.py           # Unit tests or model evaluation logic

    ├── tokenizer/
    │   ├── train_tokenizer.py      # Trains a custom tokenizer on raw corpus
    │   └── test_tokenizer.py       # Validates tokenization and decoding

    └── train/
        ├── loop.py                 # Core training loop logic (epochs, backprop, logging)
        └── optimizer.py            # Optimizer and scheduler setup (AdamW, warmup, etc.)
  



## Monitor with Weights & Biases

-Before running training, login to Weights & Biases: `wandb login`

## Training on Single GPU
    python src/main.py
  

This will:

Load the custom GPT-2 model with chosen configuration

Load 25% of the Project_Gutenberg_Australia dataset

Begin training with full logging to Weights & Biases

Save checkpoints after every epoch
