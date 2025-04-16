import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from config import TrainingConfig
from torch.nn.utils.rnn import pad_sequence

class PGSDataset(Dataset):
    """
    Project Gutenberg Dataset Loader
    
    Features:
    - Automatically handles 25% data sampling
    - Returns token sequences with attention masks
    
    Methods:
        __init__: Load dataset from Hugging Face hub
        __len__: Return dataset size
        __getitem__: Get individual sample
    """
    def __init__(self, split="train", sample_fraction=1.0):
        self.dataset = load_dataset(
            "SouthernCrossAI/Project_Gutenberg_Australia",
            split=f"{split}[:{int(sample_fraction * 100)}%]"
        )
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert token sequence to tensor
        token_ids = self.dataset[idx]["cl100k_base"]
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(token_ids), dtype=torch.long)
        }
    

def collate_fn(batch):
    """Dynamic padding for variable-length sequences"""
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    
    # Pad sequences to longest in batch
    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        'input_ids': padded_ids,
        'attention_mask': padded_masks
    }

def get_dataloader(batch_size=8, sample_fraction=1.0):
    """
    Create configured DataLoader for training
    
    Args:
        batch_size: Samples per batch
        
    Returns:
        DataLoader: Configured with:
        - Automatic batching
        - Dynamic padding
        - Shuffling (single-GPU mode)
        - Distributed sampler (multi-GPU mode)
    """
    dataset = PGSDataset(sample_fraction=sample_fraction)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )