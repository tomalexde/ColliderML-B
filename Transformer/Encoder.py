from common_imports import *
from Transformer.MultiHeadAttention import MultiHeadAttention
from typing import Optional


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer for dense padded tensors.

    Accepts a key_padding_mask (B, seq_len) bool tensor — True = pad position —
    and passes it through to MultiHeadAttention so padded hit positions are
    correctly ignored during self-attention.
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

        self.linear1  = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model,  bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.activation = activation

    def _sa_block(self, x, key_padding_mask):
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, key_padding_mask=None):
        """
        Args:
            src:              (B, seq_len, d_model)
            key_padding_mask: (B, seq_len) bool, True = pad (ignored in attention)
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x