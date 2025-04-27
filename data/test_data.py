from datasets import load_dataset
from torch.utils.data import DataLoader

def Test_Dataloaders(config):
    print("---------Starting Data Test!----------")
    dataset_in = config.data.dataset_in
    batch_size = config.data.batch_size
    columns = config.data.columns
    shuffle = config.data.shuffle

    print(f"-Loading Dataset {dataset_in} From HuggingFace!-")
    dataset = load_dataset(dataset_in, split=None)
    dataset = dataset.with_format("torch", columns=columns)

    print("----------Creating Dataloaders!----------")
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=False) if 'validation' in dataset else None
    test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False) if 'test' in dataset else None
    print("----------Dataloaders Done!----------")

    print("\n--- Testing a batch from each dataloader ---")
    batch = next(iter(train_loader))
    print(f"Train batch - input_ids shape: {batch['input_ids'].shape}")

    if val_loader is not None:
        batch = next(iter(val_loader))
        print(f"Validation batch - input_ids shape: {batch['input_ids'].shape}")

    if test_loader is not None:
        batch = next(iter(test_loader))
        print(f"Test batch - input_ids shape: {batch['input_ids'].shape}")

    print("\n--- Dataloader testing complete ---")

    return train_loader, val_loader, test_loader

