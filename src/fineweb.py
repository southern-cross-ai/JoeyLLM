import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Global variables (for child processes)
enc = None
eot = None
field = None

# Initialize tokenizer for each child process
def init_worker(enc_name, eot_token, field_name):
    global enc, eot, field
    enc = tiktoken.get_encoding(enc_name)
    eot = eot_token
    field = field_name

# Tokenize function
def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc[field]))
    return np.array(tokens, dtype=np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def main(args):
    dataset_path = args.dataset
    dataset_name = dataset_path.split("/")[-1]
    field_name = args.field
    shard_size = int(1e8)  # 100M tokens per shard

    DATA_CACHE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tokenized_data", dataset_name)
    )
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Download datasets
    fw = load_dataset(dataset_path, split="train")
    print(f"Loaded dataset: {dataset_path}, total {len(fw)} documents.")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    eot_token = tokenizer._special_tokens['<|endoftext|>']

    # Main Loop
    mp.freeze_support()
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs, initializer=init_worker, initargs=("gpt2", eot_token, field_name)) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{dataset_name}_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{dataset_name}_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize HuggingFace dataset into shards.")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset path")
    parser.add_argument("--field", type=str, default="tweet", help="Field to tokenize (default: 'tweet')")
    args = parser.parse_args()
    main(args)
