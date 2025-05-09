# test/test_model.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from types import SimpleNamespace
from src.model.joeyllm import JoeyLLM


def test_forward_pass():
    config = SimpleNamespace(
        vocab_size=100,
        max_seq_len=32,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )
    model = JoeyLLM(config)
    dummy_input = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    output = model(dummy_input)

    # Check if the output shape is correct
    assert output.shape == (2, config.max_seq_len, config.vocab_size)
