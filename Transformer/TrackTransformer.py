from common_imports import *
from Transformer.Encoder import TransformerEncoderLayer
from collections import OrderedDict


class TrackT(nn.Module):
    """
    Transformer using flash_attn_varlen_func — no padding, packed inputs.

    Input format (from DataModule collate_packed):
        x_packed   : (total_hits, 3)   all real hits concatenated
        cu_seqlens : (B+1,) int32      cumulative lengths [0, n0, n0+n1, ...]
        max_seqlen : int               longest sequence in batch

    bf16 precision is required for flash-attn. Two layers have non-%8 dims
    that would crash bf16 cuBLAS:
        - embedding:   Linear(3 → hidden_size)   [feature_dim=3, not %8]
        - classifier:  Linear(hidden_size → 4)   [output=4, not %8]
    Both are explicitly run in fp32 via autocast(enabled=False) guards.
    Everything else runs in bf16 as normal.
    """

    def __init__(
        self,
        feature_dim: int = 3,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        output_size: int = 4,
    ):
        super().__init__()
        self.name = "TrackTransformer"

        # fp32-only layers (non-%8 dimensions crash bf16 cuBLAS)
        # Stored as float32 explicitly so autocast doesn't touch them
        self.embedding   = nn.Linear(feature_dim, hidden_size).float()
        self.pos_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ).float()
        self.classifier  = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ).float()

        # bf16-safe layers (all dims are multiples of hidden_size)
        self.norm_final = nn.LayerNorm(hidden_size)
        self.layers     = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4)
            for _ in range(num_encoder_layers)
        ])

    def forward(
        self,
        x_packed:   torch.Tensor,   # (total_hits, 3)       float32 from DataModule
        cu_seqlens: torch.Tensor,   # (B+1,)  int32
        max_seqlen: int,
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, output_size)
        """
        B = cu_seqlens.shape[0] - 1

        # ------------------------------------------------------------------
        # Step 1 — embed coordinates
        # Run in fp32 (autocast disabled) because Linear(3→hidden) has k=3
        # which is not a multiple of 8 — bf16 cuBLAS would crash.
        # ------------------------------------------------------------------
        with torch.autocast(device_type='cuda', enabled=False):
            x = self.embedding(x_packed.float()) + self.pos_encoder(x_packed.float())

        # Cast to bf16 for the transformer layers
        x = x.to(torch.bfloat16)

        # ------------------------------------------------------------------
        # Step 2 — transformer encoder (packed, no padding)
        # flash-attn handles attention. FFN and LayerNorm run in bf16.
        # ------------------------------------------------------------------
        for layer in self.layers:
            x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.norm_final(x)  # (total_hits, hidden_size)

        # ------------------------------------------------------------------
        # Step 3 — per-event pooling on packed tensor
        # cu_seqlens tells us where each event's hits are in the packed tensor
        # ------------------------------------------------------------------
        x_fp32 = x.float()   # back to fp32 for pooling + classifier

        avg_pool = torch.zeros(B, x.shape[-1], device=x.device, dtype=torch.float32)
        max_pool = torch.full((B, x.shape[-1]), float('-inf'), device=x.device, dtype=torch.float32)

        for i in range(B):
            start = cu_seqlens[i].item()
            end   = cu_seqlens[i + 1].item()
            event = x_fp32[start:end]       # (n_hits_i, hidden_size)

            avg_pool[i] = event.mean(dim=0)
            max_pool[i] = event.max(dim=0).values

        pooled = torch.cat([avg_pool, max_pool], dim=1)   # (B, hidden_size * 2)

        # ------------------------------------------------------------------
        # Step 4 — classify in fp32
        # Linear(hidden*2 → hidden) is fine in bf16 but Linear(hidden → 4)
        # has output_dim=4 which is not %8 — run whole head in fp32.
        # ------------------------------------------------------------------
        with torch.autocast(device_type='cuda', enabled=False):
            return self.classifier(pooled)