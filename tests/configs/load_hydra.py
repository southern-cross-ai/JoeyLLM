from hydra import initialize_config_dir, compose
from omegaconf import DictConfig
from pathlib import Path

def load_config(config_name="config") -> DictConfig:
    config_path = Path(__file__).parent.resolve()
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name=config_name)
    return cfg

