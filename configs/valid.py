from pydantic import BaseModel, Field
from typing import Tuple


class WandBConfig(BaseModel):
    project: str
    name: str
    mode: str = Field(..., pattern="^(online|offline|disabled)$")

class DataConfig(BaseModel):
    data_path: str
    batch_size: int
    num_workers: int
    shuffle: bool     
    drop_last: bool
    pin_memory: bool  
        

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
    accumulation_steps: int
    save_model_path: str
    log_freq: int

class SchedulerConfig(BaseModel):
    max_lr: float
    pct_start: float
    anneal_strategy: str
    div_factor: float
    final_div_factor: float
    cycle_momentum: bool
    base_momentum: float
    max_momentum: float
    three_phase: bool

class Config(BaseModel):
    wandbconfig: WandBConfig
    dataconfig: DataConfig
    modelconfig: ModelConfig
    optimizerconfig: OptimizerConfig
    schedulerconfig: SchedulerConfig
    trainconfig: TrainConfig

