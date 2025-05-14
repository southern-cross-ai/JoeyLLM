from tests.configs.load_hydra import load_config
from configs.config import JoeyConfig

def test_pydantic_config_validation():
    cfg = load_config()
    validated = JoeyConfig(**cfg)

    assert validated.model.embed_dim > 0
    assert validated.data.batch_size >= 1
    assert validated.train.learning_rate > 0

