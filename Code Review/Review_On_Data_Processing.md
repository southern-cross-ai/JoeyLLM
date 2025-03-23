# ✅ Code Review: `fineweb.py`

**Reviewer:** Haoqing Liu
**Date:** March 23, 2025
**File reviewed:** `fineweb.py` in CentralisedData repo, fineweb branch
**Purpose:** Tokenize and shard the `fineweb-edu` dataset into `.npy` files suitable for pretraining large language models.

---

### 🧾 Summary

This script downloads the [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset, tokenizes the text using the GPT-2 tokenizer (`tiktoken`), and writes the tokenized data into local `.npy` shards for use in language model pretraining.

The key features include:

- Use of `multiprocessing.Pool` for parallel tokenization
- Token-level sharding (100M tokens per shard)
- GPT-2-compatible formatting (`<|endoftext|>` delimiting)
- Automatic directory creation and final shard writing

---

### ✅ Strengths

| Aspect                            | Comment                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| **Clarity**                 | The logic is easy to follow with good variable names and inline comments.                |
| **Efficiency**              | Uses `multiprocessing` and batching (`chunksize=16`) to improve throughput.          |
| **Token Safety**            | Includes validation for token value range (`uint16` check) to prevent overflow issues. |
| **Modular Design**          | Breaks out tokenization and saving into reusable functions.                              |
| **Progress Tracking**       | Integrates `tqdm` to show progress per shard, which is helpful for long runs.          |
| **Practical Output Format** | Saves files in NumPy format for fast future loading.                                     |

---

### ✏️ Suggested Improvements

| Area                                | Issue                                                                                                              | Recommendation                                                                                                               |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Type Annotations**          | None of the functions are typed.                                                                                   | Add type hints to `tokenize(doc: dict) -> np.ndarray` and `write_datafile(filename: str, tokens_np: np.ndarray) -> None` |
| **Magic Numbers**             | `shard_size = int(1e8)` and directory names are hardcoded.                                                       | Make `shard_size` and `local_dir` configurable via `argparse`                                                          |
| **Hardcoded splits**          | Only `val` and `train` based on `shard_index == 0`.                                                          | Consider making this more explicit or flexible (e.g. val/train ratio or seed)                                                |
| **No logging on shard write** | No indication when files are written.                                                                              | Add a `print()` or logging statement when each shard is saved                                                              |
| **Tokenizer initialization**  | `enc = tiktoken.get_encoding("gpt2")` works, but GPT-2 doesn't have a special EOT token defined in `tiktoken`. | Use `tiktoken.encoding_for_model("gpt2")` and define EOT manually if needed                                                |
| **Error handling**            | No exception handling for failed tokenization or file I/O.                                                         | Add try-except around critical I/O and multiprocessing sections                                                              |
| **Final progress bar state**  | Progress bar may stay open if `token_count + len(tokens) == shard_size` exactly.                                 | Explicitly close the progress bar after each shard write (`progress_bar.close()`)                                          |

---

### ✅ Review Checklist (Rubric Alignment)

| Criteria                                   | Status | Notes                                                           |
| ------------------------------------------ | ------ | --------------------------------------------------------------- |
| Completed task with clear output           | ✅     | Successfully downloads, tokenizes, and writes data              |
| Code structure is clean and logical        | ✅     | Clear separation of concerns                                    |
| Progress and results are traceable         | ✅     | Includes progress bar and shard naming convention               |
| Opportunities for reflection present       | ✅     | Comments and structure make future improvements easy            |
| Ready for integration with larger pipeline | ✅     | Data format and shard structure compatible with LLM pretraining |

---

### 🏁 Final Remarks

This is a solid, production-ready preprocessing script for educational-scale LLM training datasets. It follows good practices in batching, multiprocessing, and token format handling. With a few minor improvements (especially parameterization and error handling), it would be ready for full-scale pipeline integration or public release.
