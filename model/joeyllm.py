import torch
import torch.nn as nn
import torch.nn.functional as F

class JoeyLLM(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_layers, num_heads, dropout):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Output head with weight tying
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        B, T = input_ids.size()

        # Embedding + position
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding[:, :T, :]
        x = token_emb + position_emb

        # Causal mask (True means masked)
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()

        # Apply transformer layers
        for layer in self.blocks:
            x = layer(x, src_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
