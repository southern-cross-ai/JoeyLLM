import os
import torch
from model import JoeyLLM
from data.dataset import get_dataloader
from train.trainer import Trainer
from utils.logger import Monitor




# export TRITON_CACHE_DIR=/tmp/triton_cache/$USER
# mkdir -p "$TRITON_CACHE_DIR"




def main():
    r0 = Monitor()    

    r0.wb("on")

    r0.print("✅ Loaded Config:")

    r0.print("📦 Loaded Dataset...")
    dataset= get_dataloader(
        data_path="sample/10BT",
        chunk_size=512,
        buffer_text_size=5000, 
        batch_size=16, 
        num_workers=4,
    )

    # Print one batch to test
    for i, batch in enumerate(dataset):
        r0.print(f"📦 Sample batch {i}:")
        if isinstance(batch, dict):
            for k, v in batch.items():
                r0.print(f"{k}: {v.shape} | dtype: {v.dtype}")
        elif isinstance(batch, (list, tuple)):
            for idx, item in enumerate(batch):
                r0.print(f"Item {idx}: {item.shape} | dtype: {item.dtype}")
        else:
            r0.print(f"Unknown batch format: {type(batch)}")
        break  # Only one batch

    
    r0.print("🧠 Initializing Model...")
    model = JoeyLLM(
        vocab_size=100256,
        max_seq_len=512,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
    )

    r0.wb("model", model=model)

    # r0.print("🚀 Launching Trainer...")
    # trainer = Trainer(
    #     model=model,
    #     dataset=dataset,
    #     logger=r0,
    # )

    r0.print("🏁 Training complete!")

    r0.wb("off")

if __name__ == "__main__":
    main()
