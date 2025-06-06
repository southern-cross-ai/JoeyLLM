import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from model import JoeyLLM
from data import get_dataloader
from utils.logger import wandbLogger
from train.trainer import  Trainer

# # this is for offline testing
# import os
# print("Script stopped no errors up to this point :) ")
# sys.exit()

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("✅ Loaded Config:")

    wandbLogger.set_mode(cfg.wandb.mode)

    logger = wandbLogger(
        project_name=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    print("📦 Loading Dataset...")
    dataloader = get_dataloader(
        data_path=cfg.data.data_path,
        chunk_size=cfg.data.chunk_size,
        buffer_text_size=cfg.data.buffer_text_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    
    print("🧠 Initializing Model...")
    model = JoeyLLM(
        vocab_size=cfg.model.vocab_size,
        max_seq_len=cfg.model.max_seq_len,
        embed_dim=cfg.model.embed_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    )
    
    logger.watch_model(model, log="all", log_freq=10)

    print("📈 Loading Optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)


    print("🚀 Launching Trainer...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        logger=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


    # # this is for offline testing
    # import os
    # print("Script stopped no errors up to this point :) ")
    # sys.exit()

    trainer.fit(
        num_epochs=1,
        checkpoint_path="checkpoints/checkpoint.pth"
    )
    
    logger.finish()

    print("✅ Training Done!")   

if __name__ == "__main__":
    main()
