import torch
import torch.nn as nn
from joey.config import ModelConfig


class JoeyLLM(nn.Module):
    """Main Joey Model."""
    def __init__(self, configs:ModelConfig):
        """
        Args:
            configs (DictConfig): Model configuration settings.
        """
        super().__init__()
        assert configs.type.lower() == "joeyllm"

        self.vocab_size = configs.vocab_size
        self.max_seq_len = configs.max_seq_len
        self.embed_dim = configs.embed_dim
        self.num_layers = configs.num_layers
        self.num_heads = configs.num_heads
        self.dropout = configs.dropout

        # Token embedding maps token IDs to vectors
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Positional embedding adds position info to tokens
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        # Stack of decoder layers (each with attention + MLP)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=self.embed_dim * 4,
                dropout=self.dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(self.num_layers)
        ])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(self.embed_dim)

        # Output projection head (vocab logits)
        self.head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight  # Weight tying

    def forward(self, input_ids):
        """
        Forward pass of the model.
        Args:
            input_ids (Tensor): Tensor of shape (batch_size, seq_len)
        Returns:
            logits (Tensor): Shape (batch_size, seq_len, vocab_size)
        """
        B, T = input_ids.size()

        # Embed tokens and positions
        token_emb = self.token_embedding(input_ids)  # (B, T, D)
        position_emb = self.position_embedding[:, :T, :]  # (1, T, D)
        x = token_emb + position_emb

        # Causal mask: prevent looking ahead
        causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device)).bool()

        # Pass through all decoder layers
        for layer in self.decoder_layers:
            x = layer(tgt=x, memory=x, tgt_mask=~causal_mask)

        # Final normalization and projection
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

