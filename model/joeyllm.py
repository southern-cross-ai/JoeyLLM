import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm -> self-attn -> residual
        x = x + self.dropout(self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=mask)[0])
        # Pre-norm -> MLP -> residual
        x = x + self.mlp(self.ln_2(x))
        return x


class JoeyLLM(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        # Stack of GPT blocks
        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight  # Weight tying

    def forward(self, input_ids):
        B, T = input_ids.size()

        # Embedding + positional encoding
        token_emb = self.token_embedding(input_ids)         # (B, T, D)
        position_emb = self.position_embedding[:, :T, :]    # (1, T, D)
        x = token_emb + position_emb                        # (B, T, D)

        # Causal mask for left-to-right attention
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))  # shape: (T, T)

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

