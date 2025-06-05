import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from model import JoeyLLM
from data import get_dataloader
from utils.logger import wandbLogger

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
    # Testing dataloader
    # for batch in dataloader:
    #     print("✅ Got batch with shape:", batch.shape)
    #     break
    
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
    
    print("🚀 Launching Trainer...")

    
    
    
    
    
    
    logger.finish()

    print("✅ Training Done!")   

if __name__ == "__main__":
    main()
