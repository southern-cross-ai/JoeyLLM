# batch_tokenize_chunk.py

import polars as pl
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    # Parameters
    parquet_path = "../JoeyData/10BT/000_00000.parquet"
    tokenizer_name = "SouthernCrossAI/JoeyLLM_Tokenizer"
    batch_size_rows = 10000  # number of rows to concatenate per batch
    block_size = 1024  # chunk size for final blocks

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Load data with Polars
    print(f"Loading parquet data from {parquet_path}...")
    df = pl.read_parquet(parquet_path, columns=["text"])
    texts = df["text"].to_list()
    print(f"Loaded {len(texts):,} rows.\n")

    # Batch-concat texts
    print(f"Batch-concatenating every {batch_size_rows} rows...")
    batches = [" ".join(texts[i:i+batch_size_rows]) for i in range(0, len(texts), batch_size_rows)]
    print(f"Created {len(batches)} batches.\n")

    # Tokenize each batch
    all_tokens = []
    print("Tokenizing batches...")
    for batch_text in tqdm(batches, desc="Tokenizing"):
        tokens = tokenizer(batch_text)["input_ids"]
        all_tokens.extend(tokens)
    print(f"Total tokens: {len(all_tokens):,}\n")

    # Chunk into block_size tokens
    print(f"Chunking into blocks of {block_size} tokens...")
    blocks = [all_tokens[i:i+block_size] for i in range(0, len(all_tokens), block_size)]
    print(f"Created {len(blocks):,} blocks.\n")

    # Example: inspect first block
    print("First block sample:", blocks[0][:10])

    # TODO: Save blocks to disk or prepare DataLoader as needed

if __name__ == "__main__":
    main()
