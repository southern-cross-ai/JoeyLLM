import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


import torch
import pytest
from model.joeyllm import JoeyLLM  # adjust path if needed


@pytest.fixture
def dummy_configs():
    class ModelConfig:
        vocab_size = 1000
        max_seq_len = 64
        embed_dim = 256
        num_layers = 2
        num_heads = 4
        dropout = 0.1

    class DataConfig:
        batch_size = 8

    return ModelConfig(), DataConfig()


def test_model_forward(dummy_configs):
    model_config, data_config = dummy_configs

    # Instantiate the model
    model = JoeyLLM(
        vocab_size=model_config.vocab_size,
        max_seq_len=model_config.max_seq_len,
        embed_dim=model_config.embed_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout,
    )

    # Generate random input tensor
    input_ids = torch.randint(
        0,
        model_config.vocab_size,
        (data_config.batch_size, model_config.max_seq_len),
    )

    # Forward pass
    logits = model(input_ids)

    # Check output shape
    assert logits.shape == (
        data_config.batch_size,
        model_config.max_seq_len,
        model_config.vocab_size,
    ), f"Expected shape {(data_config.batch_size, model_config.max_seq_len, model_config.vocab_size)}, got {logits.shape}"

