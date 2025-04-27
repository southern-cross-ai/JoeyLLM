import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig, OmegaConf
from model import JoeyLLM, Test_Model 

@hydra.main(config_path="../configs", config_name="config", version_base=None)

def main(cfg: DictConfig):
    print("----------Current config!----------")
    print(OmegaConf.to_yaml(cfg))
    print("---------Testing model!----------")
    Test_Model(cfg)
    print("---------Testing Data!----------")
    print("---------Finshed Testing!----------")

main()

