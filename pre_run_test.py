import hydra
from omegaconf import DictConfig, OmegaConf
from model import JoeyLLM, Test_Model 
from data import Test_Dataloaders

@hydra.main(config_path="../configs", config_name="config", version_base=None)

def main(cfg: DictConfig):
    print("----------Current config!----------")
    print(OmegaConf.to_yaml(cfg))
    print("---------Testing model!----------")
    Test_Model(cfg.model, cfg.data)
    print("---------Testing Data!----------")
    Test_Dataloaders(cfg.data)
    print("---------Finshed Testing!----------")

main()

