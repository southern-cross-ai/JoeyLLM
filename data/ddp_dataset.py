from datasets import load_from_disk
from torch.utils.data import DataLoader

def get_dataset(dataset_path, columns=["input_ids", "target_ids"]):
    dataset = load_from_disk(dataset_path)
    dataset.set_format("torch", columns=columns)
    return dataset

