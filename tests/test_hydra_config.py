from config.load_config import load_config
from omegaconf import DictConfig

def test_hydra_loads_config():
    cfg = load_config()

    assert isinstance(cfg, DictConfig)

    for section in ["model", "data", "train", "wandb"]:
        assert section in cfg

