import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import hydra
from omegaconf import DictConfig

from utils.logger import Monitor
from model.joeyllm import JoeyLLM
from data.dataset import get_dataloader
from train.trainer import Trainer
from configs.valid import Config

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    vcfg = Config(**cfg)

    # Per-process setup 
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Init DDP process group
    dist.init_process_group(backend="nccl", init_method="env://")

    # Set Device to GPU 
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Set print, wandb, to rank = 0 
    r0 = Monitor(
        wandb_mode= vcfg.wandbconfig.mode,
        project=vcfg.wandbconfig.project,
        name=vcfg.wandbconfig.name
    )
 
    # Start Wandb and print World Size and Rank 
    r0.wb('Start')    
    print(f'🎖️ Rank: {rank}')
    r0.print(f'🌍 World Size (GPUs): {world_size}')
    
    # Todo add configs 
    r0.print("✅ Configs Loaded...")

    # Load Dataset and Loader
    r0.print("📦 Loading Dataset...")
    dataloader = get_dataloader(
        data_path=vcfg.dataconfig.data_path,
        chunk_size=vcfg.dataconfig.chunk_size,
        buffer_text_size=vcfg.dataconfig.buffer_text_size,
        batch_size=vcfg.dataconfig.batch_size,
        num_workers=vcfg.dataconfig.num_workers,
        tokenizer_path=vcfg.dataconfig.tokenizer_path,
        dataset_name=vcfg.dataconfig.dataset_name,
        shuffle_min_buffer=vcfg.dataconfig.shuffle_min_buffer,
        shuffle_buffer_multiplier=vcfg.dataconfig.shuffle_buffer_multiplier,
        pin_memory=vcfg.dataconfig.pin_memory,
        use_fast_tokenizer=vcfg.dataconfig.use_fast_tokenizer,
        streaming=vcfg.dataconfig.streaming,
        world_size=world_size,
        rank=rank
    )
        
    # Load Model
    r0.print("🧠 Initializing Model...")
    model = JoeyLLM(
        vocab_size=vcfg.modelconfig.vocab_size,
        max_seq_len=vcfg.modelconfig.max_seq_len,
        embed_dim=vcfg.modelconfig.embed_dim,
        num_layers=vcfg.modelconfig.num_layers,
        num_heads=vcfg.modelconfig.num_heads,
        dropout=vcfg.modelconfig.dropout,
        ).to(device)

    # # Load Optimizer     
    r0.print("📈 Loading Optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=vcfg.optimizerconfig.lr, 
        betas=vcfg.optimizerconfig.betas, 
        weight_decay=vcfg.optimizerconfig.weight_decay
        )

    # Load model info into wandb 
    r0.wb("model", model=model, log="gradients", log_freq=vcfg.trainconfig.log_freq)

    # Load and start Traner Loop
    r0.print("🚀 Launching Trainer...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=r0,
        rank=rank,
        world_size=world_size,
        total_steps=vcfg.trainconfig.total_steps,
        scheduler_cfg=vcfg.schedulerconfig,
        accumulation_steps=vcfg.trainconfig.accumulation_steps,
        save_model_path=vcfg.trainconfig.save_model_path,
        log_freq=vcfg.trainconfig.log_freq,
        non_blocking=vcfg.trainconfig.non_blocking,
    )

    trainer.train(epochs=vcfg.trainconfig.epochs)
    r0.print("🏁 Training complete!")

    # Turn off wandb
    r0.wb("Stop")

    # Stop this and all Process 
    dist.destroy_process_group()
    r0.print("✅ Done")

if __name__ == "__main__":
    main()