from typing      import List
from pydantic    import BaseModel, Field, validator

# 1) Reuse your existing JoeyConfig for all the model-specific fields…
class JoeyConfig(BaseModel):
    vocab_size:   int   = Field(...,  gt=0, description="Size of the vocabulary")
    max_seq_len:  int   = Field(...,  gt=0, description="Maximum sequence length")
    embed_dim:    int   = Field(...,  gt=0, description="Embedding dimensionality")
    num_layers:   int   = Field(...,  gt=0, description="Number of transformer layers")
    num_heads:    int   = Field(...,  gt=0, description="Number of attention heads")
    dropout:      float = Field(0.1, ge=0.0, le=1.0, description="Dropout probability")

    @validator('embed_dim')
    def embed_divisible_by_heads(cls, v, values):
        heads = values.get('num_heads')
        if heads and v % heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        return v

# 2) Wrap the extra “type” key around JoeyConfig:
class ModelConfig(JoeyConfig):
    type: Literal["joeyllm"] = Field(
        ...,
        description="Which model class to instantiate (must be 'joeyllm')"
    )

# 3) Define the data-section schema:
class DataConfig(BaseModel):
    dataset_out:    str
    dataset_in:     str
    batch_size:     int
    columns:        List[str]
    format:         str
    shuffle:        bool
    use_validation: bool
    use_test:       bool

# 4) Define the train-section schema:
class TrainConfig(BaseModel):
    batch_size: int
    device:     str

# 5) Top-level config that ties them all together:
class AppConfig(BaseModel):
    model: ModelConfig
    data:  DataConfig
    train: TrainConfig
