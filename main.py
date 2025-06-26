import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.logger import Monitor
from model.joeyllm import JoeyLLM
from data.dataset import get_dataloader
from train.trainer import Trainer

def main(rank, world_size):


        # ─── Per-process ENV ─────────────────────
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # ─── Select GPU ──────────────────────────
    torch.cuda.set_device(rank)

    # ─── Init DDP process group ──────────────
    dist.init_process_group(backend="nccl", init_method="env://")
    
    r0 = Monitor(
        wandb_mode="offline",  # "online", "offline", or "disabled"
        project="JoeyLLM",
        run_name="exp1"
    )
 
    r0.wb('on')

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    print(f'🎖️ Rank: {rank}')
    r0.print(f'🌍 World Size (GPUs): {world_size}')
    
    r0.print("✅ Loaded Config...")

    r0.print("📦 Loading Dataset...")
        
    dataloader = get_dataloader(
        data_path="sample/10BT",
        chunk_size=512,
        buffer_text_size=5000,
        batch_size=32,
        num_workers=3,
        world_size=world_size,
        rank=rank
    )
        
    r0.print("🧠 Initializing Model...")
        
    model = JoeyLLM(
        vocab_size=32000,
        max_seq_len=512,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
        ).to(device)
         
    r0.print("📈 Loading Optimizer")


    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    r0.wb("model", model=model, log="gradients", log_freq=1000)

    r0.print("🚀 Launching Trainer...")

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=r0,
        rank=rank,
        world_size=world_size,
        total_steps=464652,
    )

    trainer.train(epochs=1)

    #     trainer.fit(num_epochs=1, resume_from_latest=True)

    r0.print("🏁 Training complete!")

    r0.wb("off")

    dist.destroy_process_group()
    r0.print("✅ Done")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)

    # Launch one process per GPU, each will run main(rank, world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)





