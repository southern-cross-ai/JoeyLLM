# how to use
# python chunk_gutenberg.py 1024




import os
import sys
import numpy as np
from huggingface_hub import login
from datasets import Dataset, DatasetDict, load_dataset

# STEP 0: Login to Hugging Face Hub
# login()  # Uncomment if not already authenticated in your environment

# STEP 1: Load the original tokenized dataset
dataset = load_dataset("SouthernCrossAI/Project_Gutenberg_Australia")

# STEP 2: Configuration - dynamic chunk size (default: 512)
default_chunk_size = 512
chunk_size = int(os.getenv("CHUNK_SIZE", sys.argv[1] if len(sys.argv) > 1 else default_chunk_size))

token_column = "cl100k_base"
output_dataset_name = f"SouthernCrossAI/Gutenberg-Australia-Chunks-{chunk_size}"

# STEP 3: Chunking function
def chunk_token_column(split_data, chunk_size=512, token_column="cl100k_base"):
    print(f"Processing split with {len(split_data)} rows")
    token_lists = split_data[token_column]

    flat = np.concatenate(token_lists)
    usable_len = (len(flat) // chunk_size) * chunk_size
    flat = flat[:usable_len]
    chunks = flat.reshape(-1, chunk_size)

    return Dataset.from_dict({"input_ids": chunks.tolist()})

# STEP 4: Apply chunking to all splits
processed_dataset = DatasetDict()
for split in dataset:
    print(f"\nChunking '{split}' split with chunk size {chunk_size}...")
    processed_dataset[split] = chunk_token_column(dataset[split], chunk_size, token_column)

# STEP 5: Print size info
for split in processed_dataset:
    print(f"{split} split → {len(processed_dataset[split])} chunks of size {chunk_size}")

# STEP 6: Push to Hugging Face Hub
processed_dataset.push_to_hub(output_dataset_name)

