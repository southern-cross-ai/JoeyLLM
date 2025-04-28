from datasets import load_dataset
from .dataset import Dataloaders

def Test_Dataloaders(configs):
    print("---------Starting Data Test!----------")

    # Correct function call here
    train_loader, val_loader, test_loader = Dataloaders(configs)

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
