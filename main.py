import hydra
from omegaconf import DictConfig

from model import JoeyLLM
from data import Dataloaders
from train import JoeyLLMTrainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("✅ Loaded Config:")
    print(cfg)

    print("📦 Loading Dataset...")
    train_loader, val_loader, _ = Dataloaders(
        cfg.data.dataset_in,
        cfg.data.batch_size,
        cfg.data.columns,
        cfg.data.shuffle,
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

    print("🚀 Launching Trainer...")
    trainer = JoeyLLMTrainer(cfg, model, train_loader, val_loader)
    trainer.train_single_gpu()

    print("✅ Training Done!")


if __name__ == "__main__":
    main()

