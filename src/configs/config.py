import yaml
from pydantic import BaseModel

class Config(BaseModel):
    """
    Minimal configuration schema: only validates vocab_size.
    """
    vocab_size: int  # vocabulary size

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from a YAML file and validate against the schema.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            Config: An instance populated with validated config values.
        """
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        return cls.parse_obj(raw)