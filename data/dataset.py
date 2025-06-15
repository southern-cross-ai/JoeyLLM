import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class BufferedStreamTokenChunkDataset(IterableDataset):
    def __init__(self, hf_streaming_dataset, tokenizer, chunk_size, buffer_text_size=10000):
        self.dataset = hf_streaming_dataset
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.buffer_text_size = buffer_text_size
        
    def __iter__(self):
        buffer = []
        token_buffer = []

        for example in self.dataset:
            buffer.append(example["text"])
            if len(buffer) >= self.buffer_text_size:
                
                tokenized = self.tokenizer(
                    " ".join(buffer),
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]

                token_buffer.extend(tokenized)
                buffer = []

                while len(token_buffer) >= self.chunk_size + 1:
                    input_ids = token_buffer[:self.chunk_size]
                    target_ids = token_buffer[1:self.chunk_size + 1]

                    yield {
                        "inputs": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(target_ids, dtype=torch.long)
                    }

                    token_buffer = token_buffer[self.chunk_size:]

        # Final flush
        if buffer:
            tokenized = self.tokenizer(
                " ".join(buffer),
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            token_buffer.extend(tokenized)
        
        while len(token_buffer) >= self.chunk_size + 1:
            input_ids = token_buffer[:self.chunk_size]
            target_ids = token_buffer[1:self.chunk_size + 1]

            yield {
                "inputs": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(target_ids, dtype=torch.long)
            }   

            token_buffer = token_buffer[self.chunk_size:]

def get_dataloader(
    data_path, 
    chunk_size, 
    buffer_text_size, 
    batch_size, 
    num_workers,
    world_size: int = 1, rank: int = 0
    ):

    tokenizer = AutoTokenizer.from_pretrained("SouthernCrossAI/JoeyLLM_Tokenizer", use_fast=True)

    buffer_size = max(10_000, buffer_text_size * 5)

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        data_dir=data_path,
        split="train",
        streaming=True
    ).shuffle(buffer_size=buffer_size)

    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)

    token_dataset = BufferedStreamTokenChunkDataset(
        hf_streaming_dataset=dataset,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        buffer_text_size=buffer_text_size
    )

    dataloader = DataLoader(
        token_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
