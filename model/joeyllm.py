import torch
import torch.nn as nn

class JoeyLLM(nn.Module):
    """Main Joey Model."""
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_layers, num_heads, dropout=0.1):
        """
            Args:
            vocab_size (int): Number of tokens in the vocabulary.
            max_seq_len (int): Maximum sequence length.
            embed_dim (int): Embedding dimension size.
            num_layers (int): Number of transformer decoder layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()

        # Token embedding maps token IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding adds position info to tokens
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        # Stack of decoder layers (each with attention + MLP)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(embed_dim)

        # Output projection head (vocab logits)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
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
