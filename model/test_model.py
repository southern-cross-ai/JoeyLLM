import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model import JoeyLLM
from config import config 

def test_model():
    cfg = config.model

    model = JoeyLLM(
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout
    )

    input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len))
    logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")  # Should be (batch_size, seq_len, vocab_size)

    assert logits.shape == (cfg.batch_size, cfg.max_seq_len, cfg.vocab_size), "Output shape is incorrect!"

if __name__ == "__main__":
    test_model()

