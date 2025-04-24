from datasets import load_dataset

# Load the dataset
dataset = load_dataset("SouthernCrossAI/Project_Gutenberg_Australia", split="train")

# View a few entries
print(dataset[0])

