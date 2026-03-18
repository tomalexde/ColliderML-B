from common_imports import *
from Transformer.MultiHeadAttention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """
    Encoder layer for packed (flash-attn) inputs.
    Takes (total_hits, d_model) instead of (B, seq, d_model).
    cu_seqlens and max_seqlen are threaded through to MultiHeadAttention.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation=torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = MultiHeadAttention(
            Feature_q=d_model, Feature_k=d_model, Feature_v=d_model,
            Feature_total=d_model, nheads=nhead,
            dropout=dropout, bias=bias, **factory_kwargs,
        )
        self.linear1    = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, d_model,  bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1      = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2      = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.activation = activation

    def _sa_block(self, x, cu_seqlens, max_seqlen):
        x = self.self_attn(x, x, x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, cu_seqlens, max_seqlen):
        """
        Args:
            src:        (total_hits, d_model)  packed
            cu_seqlens: (B+1,) int32
            max_seqlen: int
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), cu_seqlens, max_seqlen)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, cu_seqlens, max_seqlen))
            x = self.norm2(x + self._ff_block(x))
        return x