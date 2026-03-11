from common_imports import *

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        Feature_q (int): Size of embedding dim for query
        Feature_k (int): Size of embedding dim for key
        Feature_v (int): Size of embedding dim for value
        Feature_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """
    def __init__(
        self,
        Feature_q: int,
        Feature_k: int,
        Feature_v: int,
        Feature_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = Feature_q == Feature_k and Feature_q == Feature_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(Feature_q, Feature_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(Feature_q, Feature_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(Feature_k, Feature_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(Feature_v, Feature_total, bias=bias, **factory_kwargs)
        Feature_out = Feature_q
        self.out_proj = nn.Linear(Feature_total, Feature_out, bias=bias, **factory_kwargs)
        assert Feature_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.Feature_head = Feature_total // nheads
        self.bias = bias
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``Batch``, ``Hits_q``, ``Feature_qk``)
            key (torch.Tensor): key of shape (``Batch``, ``Hits_kv``, ``Feature_qk``)
            value (torch.Tensor): value of shape (``Batch``, ``Hits_kv``, ``Feature_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``Batch``, ``Hits_q``, ``Hits_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (Batch, Hits_t, Feature_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)


        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (Batch, Hits_t, Feature_total) -> (Batch, Hits_t, nheads, Feature_head) -> (Batch, nheads, Hits_t, Feature_head)
        query = query.unflatten(-1, [self.nheads, self.Feature_head]).transpose(1, 2)
        # (Batch, Hits_s, E_total) -> (Batch, Hits_s, nheads, Feature_head) -> (Batch, nheads, Hits_s, Feature_head)
        key = key.unflatten(-1, [self.nheads, self.Feature_head]).transpose(1, 2)
        # (Batch, Hits_s, E_total) -> (Batch, Hits_s, nheads, Feature_head) -> (Batch, nheads, Hits_s, Feature_head)
        value = value.unflatten(-1, [self.nheads, self.Feature_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (Batch, nheads, Hits_t, Feature_head)
        attn_output = F.scaled_dot_product_attention( #attn_mask is Bias
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (Batch, nheads, Hits_t, Feature_head) -> (Batch, Hits_t, nheads, Feature_head) -> (Batch, Hits_t, Feature_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (Batch, Hits_t, Feature_total) -> (Batch, Hits_t, Feature_out)
        attn_output = self.out_proj(attn_output)

        return attn_output