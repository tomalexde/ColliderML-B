from common_imports import *
from Transformer.MultiHeadAttention import MultiHeadAttention
from typing import Optional

class TransformerEncoderLayer(nn.Module):
    """
    Designed to be compatible with:
    1. Standard Padded Tensors: (Batch, Max_Hits, D_Model)
    2. Jagged/Nested Tensors: (Batch, Variable_Hits, D_Model) - Optimized for Trackers.

    Args:
        d_model: The number of expected features in the input hit embeddings.
        nhead: The number of heads in the multiheadattention models.
        dim_feedforward: The dimension of the feedforward network model.
        dropout: The dropout value (default=0.1).
        activation: The activation function of the intermediate layer (default=relu).
        norm_first: If True, uses 'Pre-LN' (Normalization before blocks). If False, 
                    uses 'Post-LN'. Pre-LN is generally more stable for deep models.
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation: nn.Module = torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Self-Attention handles the "communication" between hits
        self.self_attn = MultiHeadAttention(
            Feature_q=d_model,
            Feature_k=d_model,
            Feature_v=d_model,
            Feature_total=d_model,
            nheads=nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        
        # Feed-Forward Network handles the "processing" of each hit individually
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def _sa_block(self, x, attn_mask, is_causal):
        """Self-attention block with residual-dropout."""
        # Note: We pass 'x' for all three (Q, K, V) because this is Self-Attention
        # The hits are looking at themselves within the same event.
        x = self.self_attn(x, x, x, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x):
        """Feed-forward block with activation and residual-dropout."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, is_causal=False):
        """
        Args:
            src: (N, L, E) tensor (standard) or Nested/Jagged tensor.
            src_mask: Optional mask for attention scores.
            is_causal: If True, applies a causal mask (usually False for trackers).
        """
        x = src
        if self.norm_first:
            # Pre-LN structure: Normalize -> Block -> Add (Residual)
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-LN structure: Block -> Add -> Normalize
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x
    

class TransformerEncoder(nn.Module):
    """
    Designed to be compatible with:
    (TODO) 1. Standard Padded Tensors: (Batch, Max_Hits, D_Model)
    2. Jagged/Nested Tensors: (Batch, Variable_Hits, D_Model) - Optimized for Trackers.
    
    Args:
        encoder_layer: The class definition to instantiate.
        d_model: The number of expected features in the input hit embeddings.
        dim_feedforward: The dimension of the feedforward network model.
        nhead: The number of heads in the multiheadattention models.
        num_layers: The number of layers of multiheadattention models.
        norm: Optional final normalization layer.
        activation: The activation function of the intermediate layer (default=relu).
    """
    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward=2048,
        norm: Optional[nn.Module] = None,
        activation: nn.Module = torch.nn.functional.relu,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layers = nn.ModuleList([
            encoder_layer(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                nhead=nhead,
                activation=activation,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor):
        """
        Processes the input through the stack of layers.
        
        Args:
            src: The input hits. Supports Jagged/Nested Tensors for 
                 variable hit counts across events.
        """
        output = src
        for layer in self.layers:
            output = layer(output)
        if self.norm is not None:
            output = self.norm(output)
        return output