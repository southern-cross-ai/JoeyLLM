import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import tiktoken

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
                tokenized = self.tokenizer.encode(
                    " ".join(buffer),
                    allowed_special=self.tokenizer.special_tokens_set
                )
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
            tokenized = self.tokenizer.encode(
                " ".join(buffer),
                allowed_special=self.tokenizer.special_tokens_set
            )
            token_buffer.extend(tokenized)

        
        while len(token_buffer) >= self.chunk_size + 1:
            input_ids = token_buffer[:self.chunk_size]
            target_ids = token_buffer[1:self.chunk_size + 1]

            yield {
                "inputs": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(target_ids, dtype=torch.long)
            }   

            token_buffer = token_buffer[self.chunk_size:]

def get_dataloader(data_path, chunk_size, buffer_text_size, batch_size, num_workers):
    tokenizer = tiktoken.get_encoding("cl100k_base")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        data_dir=data_path,
        split="train",
        streaming=True
    )

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
