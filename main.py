import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from model import JoeyLLM
from data import Dataloaders
from train import OneGPUTrainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("✅ Loaded Config:")
    print(OmegaConf.to_yaml(cfg))

    if cfg.train.wandb.log:
        wandb.init(
            project=cfg.train.wandb.get("project", "default_project"),
            name=f"train-{wandb.util.generate_id()}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )

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
    trainer = OneGPUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=cfg.train.device,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        epochs=cfg.train.epochs,
        checkpoint_path=cfg.train.checkpoint_path,
        resume_from=cfg.train.resume_from,
        save_every=cfg.train.save_every,
    )
    
    trainer.train()

    if cfg.train.wandb.log:
        wandb.finish()

    print("✅ Training Done!")


if __name__ == "__main__":
    main()
