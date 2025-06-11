import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from model import JoeyLLM
from data import get_dataloader
from utils.logger import wandbLogger
from train.trainer import Trainer
from utils.distributed import init_distributed, cleanup_distributed

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rank, world_size, local_rank, distributed = init_distributed()

    try:
        if rank == 0:
            print("✅ Loaded Config:")

        wandbLogger.set_mode(cfg.wandb.mode)

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        logger = None
        if rank == 0:
            logger = wandbLogger(
                project_name=cfg.wandb.project,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        if rank == 0:
            print("📦 Loading Dataset...")
        
        dataloader = get_dataloader(
            data_path=cfg.data.data_path,
            chunk_size=cfg.data.chunk_size,
            buffer_text_size=cfg.data.buffer_text_size,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            world_size=world_size,
            rank=rank
        )
        if rank == 0:
            print("🧠 Initializing Model...")
        
        
        model = JoeyLLM(
            vocab_size=cfg.model.vocab_size,
            max_seq_len=cfg.model.max_seq_len,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
        ).to(device)

        
        if rank == 0:
            print("📈 Loading Optimizer")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
            )

        if logger:
            logger.watch_model(model, log="all", log_freq=10000)

        if rank == 0:
            print("🚀 Launching Trainer...")

        trainer = Trainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            scheduler=None,
            device=device,
            rank=rank
        )

        trainer.fit(num_epochs=1, resume_from_latest=True)


        if rank == 0:
            print("🏁 Training complete!")

        if logger:
            logger.finish()

    finally:
        if distributed:
            cleanup_distributed()

        if rank == 0:
            print("✅ Done!")


if __name__ == "__main__":
    main()
