from datasets import load_dataset
from torch.utils.data import DataLoader

def Dataloaders(configs):
    print("---------Starting Data Step!----------")
    dataset_in = configs.dataset_in
    batch_size = configs.batch_size
    columns = configs.columns
    shuffle = configs.shuffle

    print(f"-----Loading Dataset {dataset_in} From HuggingFace!-----")
    dataset = load_dataset(dataset_in, split=None)
    dataset = dataset.with_format("torch", columns=columns)

    print("----------Creating Dataloaders!----------")
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=False) if 'validation' in dataset else None
    test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False) if 'test' in dataset else None
    print("----------Dataloaders Done!----------")

    return train_loader, val_loader, test_loader

