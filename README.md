# 🧠 Web Dataset Tokenization & Processing Pipelines (FineWeb-Inspired)

This project provides scalable, modular pipelines for processing large-scale web datasets, with a primary focus on **tokenizing HuggingFace datasets for language model pretraining**.

It is **inspired by the [`examples/`](https://github.com/huggingface/datatrove/tree/main/examples) directory in [Hugging Face Datatrove](https://github.com/huggingface/datatrove)**, with customization for education, experimentation, and smaller datasets.

---

## 🔑 What's Included

```
.
├── examples/                       # FineWeb-style full pipelines (legacy)
├── src/
│   └── fineweb.py                 # 🔥 Main script: HuggingFace dataset tokenizer + sharder
├── tokenized_data/                #  Where tokenized shards are saved
├── LICENSE
└── README.md
```

---

## ✨ Main Tool: `fineweb.py` — Tokenize & Shard HuggingFace Datasets

`fineweb.py` is the main entry point for this project. It:

- Downloads any HuggingFace dataset
- Uses GPT-2 tokenizer (via `tiktoken`) to tokenize a selected text field
- Saves tokens as NumPy `.npy` shards (each ~100M tokens)
- Works cross-platform (Windows/macOS/Linux), supports multiprocessing

### ✅ How to Run

```bash
python fineweb.py --dataset <HuggingFace dataset path> [--field <fieldname>]
```

#### Example:

```bash
python fineweb.py --dataset SouthernCrossAI/Tweets_cricket --field tweet
```

- `--dataset` (**required**): the dataset path on HuggingFace
- `--field` (**optional**, default: `"tweet"`): the key of the text field to tokenize

### 📍 How to find the correct `--field` name?

Run the following in Python to inspect the structure of the dataset:

```python
from datasets import load_dataset
ds = load_dataset("SouthernCrossAI/Tweets_cricket", split="train")
print(ds[0])
```

Sample output:

```python
{
  "id": 42,
  "tweet": "Australia wins the match! 🏏",
  "location": "Melbourne"
}
```

In this case, you should use `--field tweet`.

---

### 📁 Output Structure

After running, you'll find tokenized shards in:

```
tokenized_data/
└── Tweets_cricket/
    ├── Tweets_cricket_val_000000.npy
    ├── Tweets_cricket_train_000001.npy
    └── ...
```

Each `.npy` file contains an array of `np.uint16` tokens, suitable for LLM pretraining.

---

## 🧩 (Optional) Other Pipelines

These are legacy or complementary to the main tokenizer.

### 🔁 URL Deduplication Pipeline

```bash
python url_dedup_pipeline.py <input_folder> <output_folder> <sig_folder>
```

- Detects and removes duplicate documents by URL signature
- Output in `output/` and `removed/`

---

### 🌐 FineWeb-Style SLURM Pipeline (Advanced)

For large-scale Common Crawl data (with `Trafilatura`, language filtering, MinHash dedup):

Run using SLURM-compatible environment:

```python
main_processing_executor.run()
stage4.run()
```

Requires SLURM, S3 access, and modifying variables like `DUMP_TO_PROCESS`, `MAIN_OUTPUT_PATH`.

---

## 🛠️ Requirements

```bash
pip install datasets tiktoken tqdm numpy
```

For full FineWeb pipelines: also requires `datatrove`, SLURM, and optionally AWS CLI.

---

## 🙌 Acknowledgements

Built upon the open-source work of [Hugging Face Datatrove](https://github.com/huggingface/datatrove).  
This repo is for educational, research, and non-commercial use.

### 👨‍💻 Contributors

Guangxin, Ashley, Erica, Haoqing

📜 Licensed under upstream Datatrove terms where applicable.
