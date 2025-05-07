"""
config.py

Defines application configuration schemas using Pydantic,
and provides a helper to load and validate settings from a YAML file.
"""

from pydantic import BaseModel, Field
from typing import List, Literal
import yaml

# --- Model configuration schema ---
class ModelConfig(BaseModel):
    """
    Configuration for the transformer model hyperparameters.
    """
    type: Literal["Joeyllm"]  # model type identifier
    vocab_size: int = Field(..., gt=0, description="Size of the vocabulary")  # vocabulary size
    max_seq_len: int = Field(..., gt=0, description="Max sequence length")  # max tokens per input
    embed_dim: int = Field(..., gt=0, description="Embedding dimensionality")  # embedding vector size
    num_heads: int = Field(..., gt=1, description="Attention heads")  # number of attention heads
    num_layers: int = Field(..., gt=0, description="Transformer layers")  # number of transformer blocks
    dropout: float = Field(..., ge=0, le=1, description="Dropout rate")  # dropout probability

# --- Data pipeline configuration schema ---
class DataConfig(BaseModel):
    """
    Configuration for data loading and preprocessing.
    """
    dataset_out: str  # path or identifier for output dataset
    dataset_in: str  # path or identifier for input dataset
    batch_size: int = Field(..., gt=0)  # batch size for data loader
    columns: List[str]  # list of column names to include
    format: Literal["torch", "tf", "np"]  # data format
    shuffle: bool  # whether to shuffle the dataset
    use_validation: bool  # whether to split out a validation set
    use_test: bool  # whether to split out a test set

# --- Training loop configuration schema ---
class TrainConfig(BaseModel):
    """
    Configuration for the training loop parameters.
    """
    batch_size: int = Field(..., gt=0)  # batch size used during training
    device: Literal["cpu", "gpu", "auto"] = "auto"  # device for training ('cpu', 'gpu', or 'auto')

# --- Top-level application configuration ---
class AppConfig(BaseModel):
    """
    Aggregates all configuration sections into one object.
    """
    model: ModelConfig  # model hyperparameters
    data: DataConfig  # data processing settings
    train: TrainConfig  # training settings

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """
        Load configuration from a YAML file and validate against the schema.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            AppConfig: An instance populated with validated config values.

        Raises:
            pydantic.ValidationError: If the YAML does not conform to the schema.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.parse_obj(raw)
