from pydantic import BaseModel, Field, root_validator
from typing import List
from typing import Literal



class JoeyConfig(BaseModel):
    vocab_size: int = Field(..., gt=0) # Vocabulary size
    max_seq_len: int = Field(..., gt=0) # Maximum sequence length
    embed_dim:    int = Field(..., gt=0) # Embedding dimension
    # Number of layers, heads, and dropout rate
    # These are hyperparameters for the transformer architecture
    # and should be set according to the model's needs.
    # The dropout rate is a regularization technique to prevent overfitting.
    # It should be between 0 and 1.
    # A value of 0 means no dropout, while a value of 1 means all neurons are dropped.
    # A common value for dropout is around 0.1.
    # The number of layers and heads should be positive integers.
    # The number of layers is the depth of the transformer model,
    # while the number of heads is the number of attention heads in each layer.
    # The embedding dimension should be divisible by the number of heads.
    # This is because each head will have its own set of weights,
    # and the embedding dimension needs to be split evenly among them.
    # The embedding dimension is the size of the vector representation of each token.
    num_layers:   int = Field(..., gt=0)
    num_heads:    int = Field(..., gt=0)
    dropout:      float = Field(0.1, ge=0.0, le=1.0)

    @root_validator
    def check_embed_divisible_by_heads(cls, values):
        d, h = values.get('embed_dim'), values.get('num_heads') # Unpack values
        # If d is not None and h is not None and d % h != 0:
        if d is not None and h is not None and d % h != 0:
            raise ValueError('`embed_dim` must be divisible by `num_heads`')
        return values


# 2) Wrap the extra “type” key around JoeyConfig:
class ModelConfig(JoeyConfig):
    type: Literal["joeyllm"] = Field(...,
        description="Which model class to instantiate (must be 'joeyllm')")


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