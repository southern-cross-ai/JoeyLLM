from pathlib import Path
import yaml
from pydantic import BaseModel, root_validator
import torch
from typing import List

# --- Config Models ---
class ModelConfig(BaseModel):
    type: str
    vocab_size: int
    max_seq_len: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    batch_size: int
    device: str = "auto"

    @root_validator(pre=True)
    def resolve_device(cls, values):
        if values.get("device") == "auto":
            values["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        return values

class HuggingFaceDatasetConfig(BaseModel):
    dataset_name: str
    columns: List[str] = ["input_ids"]
    format: str = "torch"
    shuffle: bool = True
    use_validation: bool = True
    use_test: bool = True

class MainConfig(BaseModel):
    model: ModelConfig
    huggingface: HuggingFaceDatasetConfig

# --- Load YAML relative to config.py ---
def _load_yaml_config(filename: str = "config.yaml") -> MainConfig:
    config_path = Path(__file__).parent / filename
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return MainConfig(**raw)

config = _load_yaml_config()

