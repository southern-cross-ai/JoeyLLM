from pydantic import BaseModel, Field
from typing import List, Tuple
from pydantic_settings import BaseSettings


class WandBConfig(BaseModel):
    name: str
    run_name: str
    wandb_mode: str = Field(..., pattern="^(online|offline|disabled)$")

class ProcessConfig(BaseModel):
    rank: int
    local_rank: int
    world_size: int

class Config(BaseModel):
    wandb: WandBConfig
    process: ProcessConfig