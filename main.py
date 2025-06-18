import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from model import JoeyLLM
from data import get_dataloader
# from train.trainer import Trainer

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

        print("✅ Loaded Config:")

        print("📦 Loaded Dataset...")
        dataloader = get_dataloader(
            data_path=cfg.data.data_path,
            chunk_size=cfg.data.chunk_size
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

        # print("🚀 Launching Trainer...")
        # trainer = Trainer(
        #     model=model,
        #     dataloader=dataloader,
        #     logger=logger,
        # )

        # print("🏁 Training complete!")


if __name__ == "__main__":
    main()
