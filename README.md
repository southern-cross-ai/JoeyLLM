# 🧠 Web Dataset Processing Pipelines (Inspired by Datatrove)

This project provides a set of data processing pipelines for large-scale web datasets.
It is **heavily inspired by the `examples/` directory in [Hugging Face Datatrove](https://github.com/huggingface/datatrove)**, with customization for educational purposes.

Our pipelines are implemented using the `datatrove` framework and cover two major workflows:

- URL-based deduplication using signature comparison
- FineWeb-style dataset processing with quality filtering and minhash deduplication

---

## 📦 Project Structure

├── examples              #Full pipeline for processing data (FineWeb style)
├── README.md                          # Project overview and usage guide
└── LICENSE

## 🔁 1. URL Deduplication Pipeline

A three-step local pipeline that:

Reads input JSONL files and generates normalized URL signatures

Detects duplicate URLs using signature matching

Filters out duplicate documents

🚀 Run locally:

```bash
python url_dedup_pipeline.py <input_folder> <base_output_folder> <sigs_dup_folder>
```

Output includes:

Deduplicated data in base_output_folder/output/

Removed duplicates in base_output_folder/removed/

## 🌐 2. FineWeb-Style SLURM Pipeline

A scalable processing pipeline for Common Crawl data, similar to FineWeb.
It performs:

### 🔹 Base Processing:

- WARC reading from S3
- URL filtering
- HTML content extraction via Trafilatura
- Language and quality filtering
- Output to cleaned JSONL

### 🔹 Minhash Deduplication:

- Minhash signature generation
- Bucketing and clustering
- Final deduplication and token counting

### 🚀 Run on SLURM:

```python

# Inside fineweb_processing_pipeline.py
main_processing_executor.run()
stage4.run()
```

Other intermediate stages (signature, bucket, cluster) run automatically via `depends=` logic.

Make sure to update the following variables before running:

- DUMP_TO_PROCESS
- MAIN_OUTPUT_PATH

## 🛠️ Requirements

Install the official datatrove library:

```bash
pip install git+https://github.com/huggingface/datatrove.git
```

Other dependencies:

- Python 3.8+
- NumPy
- SLURM (for distributed processing)

## 📚 Inspiration & Attribution

This repository is adapted from and inspired by the examples/ directory in:

## 👉 Hugging Face Datatrove GitHub Repository

We gratefully acknowledge their open-source contributions.

### 👨‍💻 Contributors

Guangxin, Ashley, Erica, Haoqing

📜 License
This repository is for educational and non-commercial use only.
Please refer to the Datatrove license for upstream licensing terms.
