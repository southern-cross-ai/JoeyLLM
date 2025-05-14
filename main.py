import hydra
from omegaconf import DictConfig
from configs import JoeyConfig

from model import JoeyLLM
from data import Dataloaders
from train import JoeyLLMTrainer

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Validate with Pydantic
    try:
        model_cfg = JoeyConfig(**cfg)
    except Exception as e:
        print("❌ Validation Error:")
        raise e

    print("✅ Pydantic Model Config:")
    print(model_cfg)

    print("Loading Dataset!")
    train_loader, val_loader, _ = Dataloaders(cfg.data)

    print("Loading Model!")
    model = JoeyLLM(cfg.model)

    # print("Loading Model to GPU!")

    print("Loading Training Script")
    trainer = JoeyLLMTrainer(cfg, model, train_loader, val_loader)
    trainer.train_single_gpu()
    print("Training Done!")

if __name__ == "__main__":
    main()

    
