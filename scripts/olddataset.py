from datasets import load_dataset
from torch.utils.data import DataLoader

def Dataloaders(dataset_in: str, batch_size: int, columns: list, shuffle: bool):
    """
    Loads dataset and returns PyTorch dataloaders.
    """
    print(f"Loading Dataset: {dataset_in}")

    dataset = load_dataset(dataset_in, split=None)
    dataset = dataset.with_format("torch", columns=columns)

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=shuffle)

    val_loader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=False) \
        if 'validation' in dataset else None

    test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False) \
        if 'test' in dataset else None

    print("✅ Dataloaders Ready")
    return train_loader, val_loader, test_loader

