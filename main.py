import os
import torch
from model import JoeyLLM
from data.dataset import get_dataset 
from train.trainer import Trainer
from utils.logger import Monitor

def main():
    r0 = Monitor()    

    r0.wb("on")

    r0.print("✅ Loaded Config:")

    r0.print("📦 Loaded Dataset...")
    dataset= get_dataset(
        data_path="sample/10BT",
        chunk_size=512
    )
    
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

    r0.print("🚀 Launching Trainer...")
    trainer = Trainer(
        model=model,
        dataset=dataset,
        logger=r0,
    )

    r0.print("🏁 Training complete!")

    r0.wb("off")

if __name__ == "__main__":
    main()
