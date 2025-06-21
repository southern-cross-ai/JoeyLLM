#!/usr/bin/env python3
# concatenate_token_chunks.py
# Run: python concatenate_token_chunks.py

from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME      = "SouthernCrossAI/JoeyLLM_Tokenizer"
DATASET_NAME    = "HuggingFaceFW/fineweb"
DATA_DIR        = "sample/10BT"        # adjust if you point elsewhere
SPLIT           = "train"
CHUNK_SIZE      = 512
BATCH_SIZE      = 32
NUM_WORKERS     = 8

# ─── Tokeniser ───────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# ─── Load dataset (non-streaming) ────────────────────────────────────────────
hf_ds = load_dataset(
    DATASET_NAME,
    data_dir=DATA_DIR,
    split=SPLIT,
    streaming=False       # keep everything in memory; you have the RAM
)

# ─── Dataset class ───────────────────────────────────────────────────────────
class ConcatenatedTokenChunkDataset(Dataset):
    """
    • Joins every text sample into one long string.
    • Tokenises once.
    • Exposes fixed-length (chunk_size) slices of token IDs.
    """
    def __init__(self, hf_dataset, tokenizer, chunk_size: int = 512):
        self.chunk_size = chunk_size

        print("→ Concatenating text …")
        joined_text = " ".join(hf_dataset["text"])
        print(f"   Characters: {len(joined_text):,}")

        print("→ Tokenising …")
        enc = tokenizer(
            joined_text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        self.tokens = torch.tensor(enc["input_ids"], dtype=torch.long)

        self.num_chunks = len(self.tokens) // self.chunk_size
        print(f"   Total tokens:  {len(self.tokens):,}")
        print(f"   {self.chunk_size}-token chunks: {self.num_chunks:,}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        s = idx * self.chunk_size
        e = s + self.chunk_size
        return self.tokens[s:e]

# ─── Instantiate dataset & loader ────────────────────────────────────────────
dataset = ConcatenatedTokenChunkDataset(hf_ds, tokenizer, CHUNK_SIZE)

def collate(batch):
    # Each item already length-aligned → just stack
    return {"input_ids": torch.stack(batch)}

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate,
)

# ─── Smoke-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for i, batch in enumerate(loader):
        print("Batch shape:", batch["input_ids"].shape)  # (BATCH_SIZE, CHUNK_SIZE)
        break
