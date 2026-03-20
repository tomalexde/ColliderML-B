from common_imports import *
from collections import OrderedDict

try:
    from flash_attn import flash_attn_varlen_func        # FA2 (preferred on A100)
except ImportError:
    from flash_attn_interface import flash_attn_varlen_func  # FA3 (H100)


class MultiHeadAttention_Flash(nn.Module):
    """
    Multi-head attention using flash_attn_varlen_func.

    The model runs entirely in fp32. Only q, k, v are cast to bf16
    immediately before the FA call and the output is cast back to fp32.
    This means:
      - Linear(3→hidden) runs in fp32         → no %8 restriction
      - Linear(hidden→4) runs in fp32         → no %8 restriction
      - FA GEMMs run in bf16                  → all dims are hidden_size (%8 safe)
      - No dimension padding (F.pad) anywhere

    Use precision='32-true' in the Trainer.
    """

    def __init__(self, d_model, nheads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % nheads == 0
        self.nheads   = nheads
        self.head_dim = d_model // nheads
        self.dropout  = dropout
        self.q_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, cu_seqlens, max_seqlen):
        """
        x:          (total_hits, d_model)   fp32, packed
        cu_seqlens: (B+1,) int32
        max_seqlen: int
        Returns:    (total_hits, d_model)   fp32
        """
        total = x.shape[0]

        # Project in fp32 (all dims are d_model, %8 safe — but fp32 anyway)
        q = self.q_proj(x).view(total, self.nheads, self.head_dim)
        k = self.k_proj(x).view(total, self.nheads, self.head_dim)
        v = self.v_proj(x).view(total, self.nheads, self.head_dim)

        # Cast to bf16 only for the FA kernel — this is the only bf16 op
        # All dims here are head_dim = d_model/nheads, always %8 if d_model is
        out_bf16 = flash_attn_varlen_func(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            cu_seqlens_q = cu_seqlens,
            cu_seqlens_k = cu_seqlens,
            max_seqlen_q = max_seqlen,
            max_seqlen_k = max_seqlen,
            #dropout_p    = self.dropout if self.training else 0.0,
            causal       = False,
        )   # (total, nheads, head_dim) bf16

        # Cast back to fp32, merge heads, project
        out = out_bf16.to(torch.float32).view(total, self.nheads * self.head_dim)
        return self.out_proj(out)


class EncoderLayer_Flash(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout=0.1, norm_first=True):
        super().__init__()
        self.self_attn  = MultiHeadAttention_Flash(d_model, nheads, dropout)
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.norm_first = norm_first

    def _sa_block(self, x, cu_seqlens, max_seqlen):
        return self.dropout1(self.self_attn(x, cu_seqlens, max_seqlen))

    def _ff_block(self, x):
        return self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x)))))

    def forward(self, x, cu_seqlens, max_seqlen):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), cu_seqlens, max_seqlen)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, cu_seqlens, max_seqlen))
            x = self.norm2(x + self._ff_block(x))
        return x


class TrackT_Flash(nn.Module):
    """
    TrackTransformer — FlashAttention version.

    Uses packed tensors (no padding) from PackedDataModule.
    Model runs in fp32. Only the attention kernel runs in bf16 internally.
    Use precision='32-true' in the Trainer.

    No feature dimension padding required.
    Linear(3→hidden) and Linear(hidden→4) run in fp32 — no restrictions.

    Input from PackedDataModule:
        x_packed   : (total_hits, 3)   float32
        cu_seqlens : (B+1,)            int32
        max_seqlen : int
    """

    def __init__(self, feature_dim=3, hidden_size=256, num_heads=8,
                 num_encoder_layers=6, output_size=4, dropout=0.1):
        super().__init__()
        self.name = "TrackT_Flash"

        # fp32 throughout — no restrictions on any dimension
        self.embedding   = nn.Linear(feature_dim, hidden_size)
        self.pos_encoder = nn.Sequential(OrderedDict([
            ("proj", nn.Linear(feature_dim, hidden_size)),
            ("norm", nn.LayerNorm(hidden_size)),
            ("relu", nn.ReLU()),
            ("out",  nn.Linear(hidden_size, hidden_size)),
        ]))
        self.layers = nn.ModuleList([
            EncoderLayer_Flash(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.norm_final = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x_packed, cu_seqlens, max_seqlen):
        """
        x_packed:   (total_hits, 3)   all real hits packed end-to-end
        cu_seqlens: (B+1,) int32      cumulative hit counts
        max_seqlen: int               longest event in batch
        Returns: logits (B, output_size)
        """
        B = cu_seqlens.shape[0] - 1

        # Embed in fp32 — Linear(3→hidden), no %8 issue in fp32
        x = self.embedding(x_packed) + self.pos_encoder(x_packed)

        # Transformer layers — attention runs in bf16 internally, rest in fp32
        for layer in self.layers:
            x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.norm_final(x)   # (total_hits, hidden_size)

        # Per-event pooling using cu_seqlens to slice packed tensor
        avg_pool = torch.zeros(B, x.shape[-1], device=x.device)
        max_pool = torch.full((B, x.shape[-1]), float('-inf'), device=x.device)

        for i in range(B):
            start = cu_seqlens[i].item()
            end   = cu_seqlens[i + 1].item()
            event = x[start:end]
            avg_pool[i] = event.mean(dim=0)
            max_pool[i] = event.max(dim=0).values

        pooled = torch.cat([avg_pool, max_pool], dim=1)   # (B, hidden_size * 2)

        # Classify in fp32 — Linear(hidden→4), no %8 issue in fp32
        return self.classifier(pooled)