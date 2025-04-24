from datasets import load_dataset
import numpy as np

# Load the dataset
dataset = load_dataset("SouthernCrossAI/Project_Gutenberg_Australia")

def merge_tokens(dataset):
    """
    Merge all 'cl100k_base' tokens using NumPy for high speed.
    """
    all_tokens = np.concatenate(dataset["cl100k_base"]).tolist()
    print(f"Total tokens merged: {len(all_tokens):,}")
    return all_tokens

merged_train = merge_tokens(dataset["train"])
merged_validation = merge_tokens(dataset["validation"])
