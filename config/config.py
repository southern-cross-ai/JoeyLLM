import torch
from pydantic import BaseModel, Field
from typing import Optional

class ModelConfig(BaseModel):
    type: str = "Joeyllm"
    vocab_size: int = 50257
    max_seq_len: int = 64
    embed_dim: int = 768
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    batch_size: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
 
class TokenizerConfig(BaseModel):
    type: str = "custom"
    vocab_file: str
    merges_file: str
    pad_token: str = "<|pad|>"
    eos_token: str = ""

class TrainingConfig(BaseModel):
    batch_size: int
    micro_batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_accumulation_steps: int
    lr_scheduler: str
    max_grad_norm: float
    log_interval: int
    save_interval: int
    eval_interval: int
    checkpoint_dir: str

class DataConfig(BaseModel):
    train_path: str
    val_path: str
    block_size: int
    num_workers: int = 2

class DeviceConfig(BaseModel):
    type: str = "cuda"
    use_amp: bool = True

class LoggingConfig(BaseModel):
    use_wandb: bool = False
    wandb_project: Optional[str] = "joey-llm"
    output_dir: str = "logs"

class FullConfig(BaseModel):
    model: ModelConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig
    data: DataConfig
    device: DeviceConfig
    logging: LoggingConfig
    seed: int = 42

