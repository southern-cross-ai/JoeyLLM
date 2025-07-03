from pydantic import BaseModel, Field
from typing import List, Tuple
from pydantic_settings import BaseSettings


class WandBConfig(BaseModel):
    project: str
    name: str
    mode: str = Field(..., pattern="^(online|offline|disabled)$")

class DataConfig(BaseModel):
    data_path: str
    chunk_size: int
    buffer_text_size: int
    batch_size: int
    num_workers: int
        

class ModelConfig(BaseModel):
    vocab_size: int
    max_seq_len: int
    embed_dim: int
    num_layers: int
    num_heads: int
    dropout: float

class OptimizerConfig(BaseModel):
    lr: float 
    betas: Tuple[float, float]
    weight_decay: float


class TrainConfig(BaseModel):
    total_steps: int 
    epochs: int

class Config(BaseModel):
    wandbconfig: WandBConfig
    dataconfig: DataConfig
    modelconfig: ModelConfig
    optimizerconfig: OptimizerConfig
    trainconfig: TrainConfig