import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
from src import JoeyLLM
from src import Dataloaders

@hydra.main(config_path="src/configs", config_name="config", version_base=None)

def main(cfg: DictConfig):
    
    print("Loading Configs!")
    print(OmegaConf.to_yaml(cfg))

    print("Loading Dataset!")
    train_loader, val_loader, _ = Dataloaders(cfg.data)

    print("Loading Model!")
    
    print("Loading Model to GPU!")

    print("Loading Training Script")
    
    print("Training Done!")

if __name__ == "__main__":
    main()

