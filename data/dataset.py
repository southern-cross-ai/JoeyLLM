from datasets import load_dataset
from torch.utils.data import DataLoader

def get_dataloaders(config):
    dataset_in = config.data.dataset_in
    batch_size = config.data.batch_size
    columns = config.data.columns
    shuffle = config.data.shuffle

    dataset = load_dataset(dataset_in)
    dataset.set_format(type='torch', columns=columns)

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size) if 'validation' in dataset else None
    test_loader = DataLoader(dataset['test'], batch_size=batch_size) if 'test' in dataset else None

    return train_loader, val_loader, test_loader
