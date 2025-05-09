from hydra import initialize, compose
from omegaconf import DictConfig
from pathlib import Path

def load_config(config_name: str = "config") -> DictConfig:
    # Resolve absolute path to this file's directory
    config_dir = Path(__file__).resolve().parent

    # Resolve the path relative to the working directory (pytest / CLI / Docker)
    rel_path = config_dir.relative_to(Path.cwd())

    # Pass the relative path to Hydra
    with initialize(config_path=str(rel_path), version_base=None):
        cfg = compose(config_name=config_name)
    return cfg

