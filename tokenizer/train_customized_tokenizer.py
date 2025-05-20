from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
from tokenizers import normalizers
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
import os

# Define the list of datasets from Hugging Face
dataset_names = [
    "SouthernCrossAI/Project_Gutenberg_Australia",
    # "SouthernCrossAI/Wiki-Australia",
    # "SouthernCrossAI/Reddit-posts",
]

# Extract raw text from each dataset (default field is "Paragraph Text")
def load_all_texts(dataset_list, field="Paragraph Text", max_per_dataset=10000):
    all_texts = []
    for name in dataset_list:
        print(f" Loading: {name}")
        try:
            ds = load_dataset(name, split="train")
            texts = [x[field] for x in ds.select(range(min(len(ds), max_per_dataset)))]
            all_texts.extend(texts)
        except Exception as e:
            print(f" Failed to load {name}: {e}")
    print(f"Total samples collected: {len(all_texts)}")
    return all_texts

# Train the tokenizer using all collected Australian texts
def train_tokenizer(texts, vocab_size=30000, save_dir="my_tokenizer"):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFKC()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(tqdm(texts), trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ]
    )

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(f"{save_dir}/tokenizer.json")
    print(f"✅ Tokenizer saved to {save_dir}/tokenizer.json")


if __name__ == "__main__":
    all_texts = load_all_texts(dataset_names)
    train_tokenizer(all_texts)
