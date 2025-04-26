import torch
from model.joeyllm import JoeyLLM  # Make sure this is there

def Test_Model(cfg):
    model = JoeyLLM(
        vocab_size=cfg.model.vocab_size,
        max_seq_len=cfg.model.max_seq_len,
        embed_dim=cfg.model.embed_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout
    )

    input_ids = torch.randint(0, cfg.model.vocab_size, (cfg.model.batch_size, cfg.model.max_seq_len))
    logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")

    assert logits.shape == (cfg.model.batch_size, cfg.model.max_seq_len, cfg.model.vocab_size), "Output shape is incorrect!"

