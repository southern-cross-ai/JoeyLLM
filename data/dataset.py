from JoeyLLM.config.config import config
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=None):
    dataset_name = config.dataset_loader.dataset_name
    batch_size = batch_size or config.dataset_loader.batch_size

    dataset = load_dataset(dataset_name)
    dataset.set_format(type='torch', columns=['input_ids'])

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size)
    test_loader = DataLoader(dataset['test'], batch_size=batch_size)

    return train_loader, val_loader, test_loader

# ---------------------------------------------
# 🔧 NOTE FOR FUTURE DEVELOPERS:
# The block below is for **testing purposes only**.
# It will only run when this script is executed directly,
# and will NOT run if this file is imported as a module.
# ---------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

    # Sample loop to check one training batch
    for batch in train_loader:
        input_ids = batch['input_ids']
        print(f"✅ Sample batch shape: {input_ids.shape}")
        break

