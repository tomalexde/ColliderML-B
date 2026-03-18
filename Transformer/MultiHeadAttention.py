from common_imports import *
from flash_attn import flash_attn_varlen_func


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention using flash_attn_varlen_func.
 
    Takes PACKED inputs (no padding) + cu_seqlens instead of a mask.
    Requires bf16. Works on Ampere+ GPUs (A100, H100, RTX 30xx+).
 
    Memory: O(n) instead of O(n²) — flash-attn tiles the computation and
    never materialises the full attention matrix.
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
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
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
        #query = query.unflatten(-1, [self.nheads, self.Feature_head]).transpose(1, 2).contiguous()
        #key   = key.unflatten(  -1, [self.nheads, self.Feature_head]).transpose(1, 2).contiguous()
        #value = value.unflatten(-1, [self.nheads, self.Feature_head]).transpose(1, 2).contiguous()

        # Reshape for flash-attn: (total_hits, nheads, head_dim)
        # flash-attn expects this layout, NOT (B, nheads, seq, head_dim)
        total_hits = q.shape[0]
        q = q.view(total_hits, self.nheads, self.Feature_head)
        k = k.view(total_hits, self.nheads, self.Feature_head)
        v = v.view(total_hits, self.nheads, self.Feature_head)

        # flash_attn_varlen_func:
        #   - cu_seqlens tells it where each event starts/ends
        #   - attention is automatically blocked at event boundaries
        #   - no padding mask needed — pad positions don't exist
        #   - returns (total_hits, nheads, head_dim)
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q = cu_seqlens,
            cu_seqlens_k = cu_seqlens,
            max_seqlen_q = max_seqlen,
            max_seqlen_k = max_seqlen,
            dropout_p    = self.dropout if self.training else 0.0,
            causal       = False,
        )
 
        # Merge heads: (total_hits, nheads, head_dim) -> (total_hits, Feature_total)
        out = out.view(total_hits, self.nheads * self.Feature_head)
        return self.out_proj(out)