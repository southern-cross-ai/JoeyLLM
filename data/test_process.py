# test_process.py

import time
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer
import pandas as pd
import polars as pl

if __name__ == "__main__":
    # Start overall timer
    overall_start = time.time()

    # Load tokenizer
    print("Loading tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("SouthernCrossAI/JoeyLLM_Tokenizer", use_fast=True)
    print(f"Tokenizer loaded in {time.time() - start:.2f} seconds.\n")

    # Load parquet file using pandas
    print("Loading parquet data...")
    start = time.time()
    path = "../JoeyData/10BT/000_00000.parquet"
    df = pd.read_parquet(path, columns=["text"])
    print(f"Data loaded in {time.time() - start:.2f} seconds. Shape: {df.shape}\n")

    # Convert to Hugging Face Dataset
    print("Converting to Hugging Face Dataset...")
    start = time.time()
    dataset = HFDataset.from_pandas(df)
    print(f"Conversion complete in {time.time() - start:.2f} seconds.\n")

    # Define tokenize function for batched map
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    # Tokenize with multiprocessing
    print("Starting tokenization with multiprocessing...")
    start = time.time()
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=50000,
        num_proc=45
    )
    print(f"Tokenization complete in {time.time() - start:.2f} seconds.\n")

    # Inspect or save the tokenized dataset
    print(tokenized_dataset)
    # tokenized_dataset.save_to_disk("tokenized_dataset_dir")

    print(f"Total script runtime: {time.time() - overall_start:.2f} seconds.")
