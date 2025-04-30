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
├── requirements.txt            # Python dependencies

└── src/
    ├── main.py                 # Entry point: Hydra config + training start

    ├── configs/
    │   └── config.yaml         # YAML configuration for model, training, logging

    ├── data/
    │   ├── dataset.py          # Loads and batches the training dataset
    │   ├── test_data.py        # Loads validation/test datasets
    │   └── chunk.py            # Chunks long token sequences (preprocessing)

    ├── model/
    │   ├── joeyllm.py          # GPT-2 architecture (transformer blocks, attention)
    │   └── test_model.py       # Unit tests or model verification

    ├── tokenizer/
    │   ├── train_tokenizer.py  # Trains a tokenizer from raw text corpus
    │   └── test_tokenizer.py   # Tests tokenization, decoding accuracy

    └── train/
        ├── loop.py             # Training loop (loss, steps, logging, W&B)
        └── optimizer.py        # Optimizer/scheduler setup (AdamW, warmup)




## Monitor with Weights & Biases

-Before running training, login to Weights & Biases: `wandb login`

## Training on Single GPU
    python src/main.py
  

This will:

Load the custom GPT-2 model with chosen configuration

Load 25% of the Project_Gutenberg_Australia dataset

Begin training with full logging to Weights & Biases

Save checkpoints after every epoch
