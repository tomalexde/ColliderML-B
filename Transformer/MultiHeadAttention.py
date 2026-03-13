from common_imports import *


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for dense padded tensors.

    Uses separate q/k/v projections (original packed_proj + torch.chunk was
    broken: after norm1(x), query is key is always False so the wrong branch
    fired). Accepts key_padding_mask to ignore pad hit positions.
    """

    def __init__(
        self,
        Feature_q: int,
        Feature_k: int,
        Feature_v: int,
        Feature_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads       = nheads
        self.dropout      = dropout
        self.Feature_head = Feature_total // nheads

        assert Feature_total % nheads == 0, \
            f"Feature_total ({Feature_total}) must be divisible by nheads ({nheads})"

        self.q_proj   = nn.Linear(Feature_q,     Feature_total, bias=bias, **factory_kwargs)
        self.k_proj   = nn.Linear(Feature_k,     Feature_total, bias=bias, **factory_kwargs)
        self.v_proj   = nn.Linear(Feature_v,     Feature_total, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(Feature_total, Feature_q,     bias=bias, **factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query:            (B, Hits_q,  Feature_q)
            key:              (B, Hits_kv, Feature_k)
            value:            (B, Hits_kv, Feature_v)
            key_padding_mask: (B, Hits_kv) bool — True = pad position (ignored)
            is_causal:        causal mask, False for tracker
        Returns:
            (B, Hits_q, Feature_q)
        """
        # QKV projections
        query = self.q_proj(query)
        key   = self.k_proj(key)
        value = self.v_proj(value)

        # Split heads: (B, Hits, D) -> (B, nheads, Hits, D_head)
        query = query.unflatten(-1, [self.nheads, self.Feature_head]).transpose(1, 2).contiguous()
        key   = key.unflatten(  -1, [self.nheads, self.Feature_head]).transpose(1, 2).contiguous()
        value = value.unflatten(-1, [self.nheads, self.Feature_head]).transpose(1, 2).contiguous()

        # Build additive attention bias from padding mask
        attn_bias = None
        if key_padding_mask is not None:
            # (B, Hits_kv) -> (B, 1, 1, Hits_kv), broadcast over heads and query positions
            attn_bias = torch.zeros(
                key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1],
                dtype=query.dtype, device=query.device,
            )
            attn_bias.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Merge heads and output projection: (B, nheads, Hits, D_head) -> (B, Hits, D_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        return self.out_proj(attn_output)