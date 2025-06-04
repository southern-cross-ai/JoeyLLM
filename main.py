import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from model import JoeyLLM
from data import Dataloaders
from utils.logger import WandBLogger

# this is for offline testing
import os
import wandb
os.environ["WANDB_MODE"] = "offline"

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("✅ Loaded Config:")

    logger = WandBLogger(
        project_name=cfg.WandB.project,
        name=f"train-{wandb.util.generate_id()}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    print("📦 Loading Dataset...")

    print("🧠 Initializing Model...")
    model = JoeyLLM(
        vocab_size=cfg.model.vocab_size,
        max_seq_len=cfg.model.max_seq_len,
        embed_dim=cfg.model.embed_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    )
    
    wandb.watch(model, log="all", log_freq=10)
    
    print("🚀 Launching Trainer...")

    logger.finish()

    print("✅ Training Done!")

    print("Script stopped no errors up to this point :) ")
    sys.exit()
    

if __name__ == "__main__":
    main()
