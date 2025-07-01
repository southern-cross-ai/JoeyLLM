import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.logger import Monitor
from model.joeyllm import JoeyLLM
from data.dataset import get_dataloader
from train.trainer import Trainer

def main(rank, world_size):

    # Per-process setup 
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Init DDP process group
    dist.init_process_group(backend="nccl", init_method="env://")

    # Set Device to GPU 
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Set print, wandb, to rank = 0 
    r0 = Monitor(
        wandb_mode="online",  # "online", "offline", or "disabled"
        project="JoeyLLM",
        run_name="Sunday"
    )
 
    # Start Wandb and print World Size and Rank 
    r0.wb('Start')    
    print(f'🎖️ Rank: {rank}')
    r0.print(f'🌍 World Size (GPUs): {world_size}')
    
    # Todo add configs 
    r0.print("✅ Loaded Config...")

    # Load Dataset and Loader
    r0.print("📦 Loading Dataset...")
    dataloader = get_dataloader(
        data_path="sample/10BT",
        chunk_size=512,
        buffer_text_size=8000,
        batch_size=16,
        num_workers=3,
        world_size=world_size,
        rank=rank
    )
        
    # Load Model
    r0.print("🧠 Initializing Model...")
    model = JoeyLLM(
        vocab_size=32000,
        max_seq_len=512,
        embed_dim=768,
        num_layers=24,
        num_heads=16,
        dropout=0.1,
        ).to(device)

    # Load Optimizer     
    r0.print("📈 Loading Optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    # Load model info into wandb 
    r0.wb("model", model=model, log="gradients", log_freq=1000)

    # Load and start Traner Loop
    r0.print("🚀 Launching Trainer...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=r0,
        rank=rank,
        world_size=world_size,
        total_steps=190000,
    )

    trainer.train(epochs=5)
    r0.print("🏁 Training complete!")

    # Turn off wandb
    r0.wb("Stop")

    # Stop this and all Process 
    dist.destroy_process_group()
    r0.print("✅ Done")

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    main(rank, world_size)