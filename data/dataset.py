import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

class BufferedStreamTokenChunkDataset(IterableDataset):
    def __init__(self, hf_streaming_dataset, tokenizer, chunk_size):
        self.dataset = hf_streaming_dataset
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
    def __iter__(self):
        token_buffer = []

        for example in self.dataset:
            try:
                tokens = self.tokenizer(
                    example["text"],
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
            except Exception:
                continue  # skip malformed lines

            token_buffer.extend(tokens)

            while len(token_buffer) >= self.chunk_size + 1:
                input_ids = token_buffer[:self.chunk_size]
                target_ids = token_buffer[1:self.chunk_size + 1]

                yield {
                    "inputs": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(target_ids, dtype=torch.long)
                }

                token_buffer = token_buffer[self.chunk_size:]

def get_dataset(data_path, chunk_size):
    tokenizer = AutoTokenizer.from_pretrained(
        "SouthernCrossAI/JoeyLLM_Tokenizer", use_fast=True
    )

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        data_dir=data_path,
        split="train",
        streaming=True
    )

    return BufferedStreamTokenChunkDataset(
        hf_streaming_dataset=dataset,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
    )
