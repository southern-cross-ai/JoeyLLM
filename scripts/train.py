import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

@hydra.main(config_path="../configs", config_name="config", version_base=None)

def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


main()

