from pydantic import BaseModel
from typing import List, Optional

# --- Model Config ---
class ModelConfig(BaseModel):
    name: str
    vocab_size: int
    max_seq_len: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dropout: float

# --- Data Config ---
class DataConfig(BaseModel):
    dataset_out: str
    dataset_in: str
    batch_size: int
    columns: List[str]
    format: str
    shuffle: bool
    use_validation: bool
    use_test: bool

# --- Training Config ---
class TrainConfig(BaseModel):
    batch_size: int
    device: str
    epochs: int
    learning_rate: float
    weight_decay: float
    save_every: int
    resume_from: Optional[str] = None
    gradient_accumulation_steps: int
    checkpoint_path: str

# --- Top-level config ---
class JoeyConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig

