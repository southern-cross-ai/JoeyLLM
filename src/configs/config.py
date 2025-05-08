import yaml
from pydantic import BaseModel

class Config(BaseModel):
    """
    Minimal configuration schema: only validates vocab_size.
    """
    name: str
    vocab_size: int
    max_seq_len: in
    embed_dim: int 
    num_heads: int
    num_layers: int
    dropout: float

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from config.yaml 
        """
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        return cls.parse_obj(raw)
