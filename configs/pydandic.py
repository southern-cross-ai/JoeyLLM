from pydantic import BaseModel, Field
from typing import List, Tuple

class WandBtConfig(BaseModel):
    name: str = "JoeyLLM"
    run_name: str = "Sunday"
    wandb_mode: str = Field("online", regex="^(online|offline|disabled)$")

class DatasetConfig(BaseModel):
    data_path: str
    chunk_size: int = 512
    buffer_text_size: int = 8000
    batch_size: int = 16
    num_workers: int = 3

class ModelConfig(BaseModel):
    vocab_size: int = 32000
    max_seq_len: int = 512
    embed_dim: int = 768
    num_layers: int = 24
    num_heads: int = 16
    dropout: float = 0.1

class OptimizerConfig(BaseModel):
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1

class TrainerConfig(BaseModel):
    total_steps: int = 226000
    epochs: int = 5

class Config(BaseModel):
    project: ProjectConfig
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig

