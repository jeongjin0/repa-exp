import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoisingUNet1D(nn.Module):
    """1D UNet for denoising embedding vectors"""
    def __init__(self, embedding_dim=512, time_dim=128, hidden_dims=[256, 512, 512]):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )
        
        # Encoder
        self.encoder = nn.ModuleList()
        dims = [embedding_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.encoder.append(nn.Sequential(
                nn.Linear(dims[i] + time_dim, dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dims[-1] + time_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.GELU()
        )
        
        # Decoder
        self.decoder = nn.ModuleList()
        dims_reversed = list(reversed(hidden_dims)) + [embedding_dim]
        for i in range(len(dims_reversed) - 1):
            # Skip connection doubles the input dimension
            in_dim = dims_reversed[i] * 2 if i > 0 else dims_reversed[i]
            self.decoder.append(nn.Sequential(
                nn.Linear(in_dim + time_dim, dims_reversed[i+1]),
                nn.LayerNorm(dims_reversed[i+1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
        
        # Final projection
        self.final = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x, t):
        """
        x: (B, embedding_dim) - noised embedding
        t: (B,) or (B, 1) - timestep
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Time embedding
        t_emb = self.time_mlp(t)  # (B, time_dim)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)  # (B, 1, time_dim) -> (B, time_dim)

        # Encoder with skip connections
        skips = []
        h = x
        for layer in self.encoder:
            h = layer(torch.cat([h, t_emb], dim=-1))
            skips.append(h)
        
        # Bottleneck
        h = self.bottleneck(torch.cat([h, t_emb], dim=-1))
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if i > 0:  # Skip connection from encoder
                h = torch.cat([h, skips[-(i+1)]], dim=-1)
            h = layer(torch.cat([h, t_emb], dim=-1))
        
        # Final output
        out = self.final(h)
        return out