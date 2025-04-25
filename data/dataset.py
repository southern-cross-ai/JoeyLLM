from datasets import load_dataset
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=8):
    """
    Loads the Hugging Face dataset and returns PyTorch DataLoaders. This function is meant to be imported and used in your training pipelineand converts the dataset to torch tensors, and wraps it in PyTorch DataLoaders.
    """
    dataset = load_dataset("SouthernCrossAI/Gutenberg-Australia-Chunks-512")
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

