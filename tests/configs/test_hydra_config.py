from tests.configs.load_hydra import load_config
from omegaconf import DictConfig

def test_hydra_loads_config():
    cfg = load_config()

    assert isinstance(cfg, DictConfig)
    for section in ["model", "data", "train"]:
        assert section in cfg

