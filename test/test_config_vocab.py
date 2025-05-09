from config import Config

def test_vocab_size_type():
    config = Config.from_yaml("config.yaml")
    assert isinstance(config.model.vocab_size, int), "vocab_size should be an integer"

