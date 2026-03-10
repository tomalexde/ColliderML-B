  
from common_imports import *
from Transformer.Encoder import *
from collections import OrderedDict

class TrackT(nn.Module):
    def __init__(self, feature_dim=3, hidden_size=256, num_heads=8, num_encoder_layers=4, output_size=4):
        super(TrackT, self).__init__()
        self.name = "TrackTransformer"
        
        # 1. Coordinate Embedding: Project (x, y, z) into high-dimensional space
        self.embedding = nn.Linear(feature_dim, hidden_size)
        
        # 2. Geometric Positional Encoding: Learns the "importance" of specific locations
        # This is a MLP that looks at the same (x, y, z) but through different weights
        self.pos_encoder = nn.Sequential(OrderedDict([
            ("geo_proj", nn.Linear(feature_dim, hidden_size)),
            ("geo_norm", nn.LayerNorm(hidden_size)),
            ("geo_relu", nn.ReLU()),
            ("geo_out", nn.Linear(hidden_size, hidden_size))
        ]))
        
        # 3. Transformer Encoder Layers
        # We use a ModuleList to pass the Nested Tensors through each layer sequentially
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, hidden_size*4) 
            for _ in range(num_encoder_layers)
        ])
        
        # 4. Classification Head
        # We pool the hits and then reduce them to our 4 class probabilities
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # *2 because we concat mean and max
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """
        Input: x is a torch.NestedTensor of shape (Batch, [Hits], 3)
        """
        # Step 1: Combine feature info and geometric info
        # Even with Nested Tensors, addition works hit-by-hit
        x_embed = self.embedding(x)
        x_pos = self.pos_encoder(x)
        x = x_embed + x_pos
        
        # Step 2: Pass through Transformer layers
        # Each hit now "sees" every other hit in its own event
        for layer in self.layers:
            x = layer(x)
            
        # Step 3: Global Pooling for Nested Tensors
        # .mean(dim=1) on a NestedTensor correctly averages only the hits present
        avg_pool = x.mean(dim=1)       # Shape: (Batch, Hidden_Size)
        
        # .max(dim=1) finds the most 'excited' feature across the event
        max_pool = x.max(dim=1)[0]     # Shape: (Batch, Hidden_Size)
        
        # Concatenate for a rich "Event Fingerprint"
        # Shape: (Batch, Hidden_Size * 2)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Step 4: Final Prediction
        return self.classifier(pooled)