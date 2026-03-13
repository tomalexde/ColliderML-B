from common_imports import *
from Transformer.Encoder import TransformerEncoderLayer
from collections import OrderedDict


class TrackT(nn.Module):
    """
    Transformer for particle track classification.
    Uses padded dense tensors + key_padding_mask (standard approach).
    No precision hacks needed — works cleanly with fp32.
    """

    def __init__(
        self,
        feature_dim: int = 3,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        output_size: int = 4,
    ):
        super().__init__()
        self.name = "TrackTransformer"

        # 1. Coordinate embedding
        self.embedding  = nn.Linear(feature_dim, hidden_size)
        self.norm_final = nn.LayerNorm(hidden_size)

        # 2. Geometric positional encoding
        self.pos_encoder = nn.Sequential(OrderedDict([
            ("geo_proj", nn.Linear(feature_dim, hidden_size)),
            ("geo_norm", nn.LayerNorm(hidden_size)),
            ("geo_relu", nn.ReLU()),
            ("geo_out",  nn.Linear(hidden_size, hidden_size)),
        ]))

        # 3. Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4)
            for _ in range(num_encoder_layers)
        ])

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, max_hits, feature_dim)  padded dense tensor
            key_padding_mask: (B, max_hits)               True = pad position
        Returns:
            logits: (B, output_size)
        """
        # Step 1 — embed hit coordinates
        x = self.embedding(x) + self.pos_encoder(x)

        # Step 2 — transformer encoder (pad positions masked out in attention)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.norm_final(x)

        # Step 3 — mask-aware global pooling
        # Zero out pad positions so they don't affect mean or max
        real_mask = (~key_padding_mask).unsqueeze(-1).to(x.dtype)  # (B, max_hits, 1)

        # Mean: sum real hits / count real hits
        avg_pool = (x * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)

        # Max: set pad positions to -inf so they never win
        max_pool = x.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1).values

        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, hidden_size * 2)

        # Step 4 — classify
        return self.classifier(pooled)