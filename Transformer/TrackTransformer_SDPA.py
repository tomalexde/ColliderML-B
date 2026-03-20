from common_imports import *
from collections import OrderedDict


class MultiHeadAttention_SDPA(nn.Module):
    """Standard multi-head attention using F.scaled_dot_product_attention."""

    def __init__(self, d_model, nheads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % nheads == 0
        self.nheads     = nheads
        self.head_dim   = d_model // nheads
        self.dropout    = dropout
        self.q_proj     = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj     = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj     = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj   = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, key_padding_mask=None):
        """
        x:                (B, seq, d_model)
        key_padding_mask: (B, seq) bool — True = pad (ignored)
        """
        B, S, _ = x.shape

        q = self.q_proj(x).unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(x).unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(x).unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)

        attn_bias = None
        if key_padding_mask is not None:
            attn_bias = torch.zeros(B, 1, 1, S, dtype=x.dtype, device=x.device)
            attn_bias.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = attn_bias,
            dropout_p = self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).flatten(-2)
        return self.out_proj(out)


class EncoderLayer_SDPA(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout=0.1, norm_first=True):
        super().__init__()
        self.self_attn  = MultiHeadAttention_SDPA(d_model, nheads, dropout)
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.norm_first = norm_first

    def _sa_block(self, x, mask):
        return self.dropout1(self.self_attn(x, key_padding_mask=mask))

    def _ff_block(self, x):
        return self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x)))))

    def forward(self, x, key_padding_mask=None):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


class TrackT_SDPA(nn.Module):
    """
    TrackTransformer — SDPA version.

    Uses padded tensors + key_padding_mask. Runs fully in fp32.
    Use with precision='32-true' in the Trainer.

    No dimension padding required — fp32 has no %8 cuBLAS restriction.
    Linear(3→hidden) and Linear(hidden→4) work fine.

    Input from PaddedDataModule:
        x    : (B, max_hits, 3)   float32, zeros at pad positions
        mask : (B, max_hits)      bool, True = pad
    """

    def __init__(self, feature_dim=3, hidden_size=256, num_heads=8,
                 num_encoder_layers=6, output_size=4, dropout=0.1):
        super().__init__()
        self.name = "TrackT_SDPA"

        # No padding needed — fp32 handles any dimension
        self.embedding   = nn.Linear(feature_dim, hidden_size)
        self.pos_encoder = nn.Sequential(OrderedDict([
            ("proj", nn.Linear(feature_dim, hidden_size)),
            ("norm", nn.LayerNorm(hidden_size)),
            ("relu", nn.ReLU()),
            ("out",  nn.Linear(hidden_size, hidden_size)),
        ]))
        self.layers = nn.ModuleList([
            EncoderLayer_SDPA(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.norm_final = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x, key_padding_mask):
        """
        x:                (B, max_hits, 3)
        key_padding_mask: (B, max_hits) bool — True = pad
        Returns: logits (B, output_size)
        """
        x = self.embedding(x) + self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.norm_final(x)

        # Mask-aware pooling
        real = (~key_padding_mask).unsqueeze(-1).to(x.dtype)
        avg_pool = (x * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        max_pool = x.masked_fill(key_padding_mask.unsqueeze(-1), float('-inf')).max(dim=1).values

        return self.classifier(torch.cat([avg_pool, max_pool], dim=1))