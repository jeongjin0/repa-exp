import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.vision_transformer import Attention
import torch.nn.functional as F
from einops import repeat, pack, unpack
from torch.cuda.amp import autocast


class DiTZeroflowintegrated_independent_t(nn.Module):
    def __init__(self, original_dit, noise_dim=384, output_noise_dim=384, a=None):
        super().__init__()
        self.dit = original_dit
        self.noise_dim = noise_dim
        self.output_noise_dim = output_noise_dim
        self.hidden_size = original_dit.pos_embed.size(-1)
        
        # Noise vector를 토큰으로 변환
        #self.noise_to_token = nn.Linear(noise_dim, self.hidden_size)
        self.token_to_noise = nn.Linear(self.hidden_size, output_noise_dim)

        if a is None:
            self.noise_to_token = nn.Linear(128, 384)
            self.noise_to_token2 = nn.Linear(128, 384)
            self.noise_to_token3 = nn.Linear(128, 384)

        #self.label_embedder = nn.Embedding(10, 384)
        #require grad false
        #self.label_embedder.weight.requires_grad = False

        # Noise 토큰 위치 embedding
        self.noise_pos_embed = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        if a is not None:
            self.noise_pos_embed2 = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)

    def forward(self, x, t1, t2=None, y=None, noise_vector=None, args=None):
            """
            기존 DiT forward에 noise 토큰만 추가
            
            Args:
                x: [B, C, H, W] - 이미지
                t1: [B] - image용 timestep  
                t2: [B] - noise용 timestep
                y: [B] - class labels (unconditional이면 None)
                noise_vector: [B, 128] - 추가할 noise vector
            """

            if t2 is None:
                t2 = t1

            x_tokens = self.dit.x_embedder(x) + self.dit.pos_embed  # [B, num_patches, hidden_size]

            t1_embed = self.dit.t_embedder(t1)                    # [B, hidden_size]
            t2_embed = self.dit.t_embedder(t2)                    # [B, hidden_size]
            
            if y is not None:
                # Conditional case
                y_embed = self.dit.y_embedder(y, self.training)     # [B, hidden_size]
                c = t1_embed + t2_embed + y_embed                   # [B, hidden_size]
            else:
                # Unconditional case
                c = t1_embed + t2_embed                             # [B, hidden_size]
            
            if type(noise_vector) is list:
                noise_tokens = [nt.unsqueeze(1) + self.noise_pos_embed for nt in noise_vector]  # 각 nt에 positional embedding 추가
                noise_tokens = torch.cat(noise_tokens, dim=1)  # [B, N, hidden_size], N = len(noise_vector)
                all_tokens = torch.cat([x_tokens, noise_tokens], dim=1)  # [B, T + N, hidden_size]
                
            else:
                raise NotImplementedError("noise_vector should be a list of tensors.")        

            features = []
            for i, block in enumerate(self.dit.blocks):
                all_tokens = block(all_tokens, c)  # [B, num_patches+1, hidden_size]
                if i in self.dit.encoder_depths:
                    features.append(all_tokens.clone())

            
            num_patches = x_tokens.size(1)
            image_tokens = all_tokens[:, :num_patches, :]  # [B, num_patches, hidden_size]
            
            image_output = self.dit.final_layer(image_tokens, c)    # [B, num_patches, patch_size^2 * out_channels]
            image_output = self.dit.unpatchify(image_output)        # [B, out_channels, H, W]
            
            noise_token_out = all_tokens[:, num_patches, :]     # [B, hidden_size]
            noise_output = self.token_to_noise(noise_token_out) # [B, output_noise_dim] - classification logits


            return image_output, noise_output, features


class DiTZeroflowintegrated_independent_multitoken(nn.Module):
    def __init__(self, original_dit, noise_dim=384, output_noise_dim=384, 
                 num_noise_tokens=16):  # CLS token을 16개 토큰으로 표현
        super().__init__()
        self.dit = original_dit
        self.num_noise_tokens = num_noise_tokens
        self.hidden_size = original_dit.pos_embed.size(-1)
        
        # Noise를 multiple tokens으로 변환
        self.noise_to_tokens = nn.Linear(noise_dim, self.hidden_size * num_noise_tokens)
        
        # Multiple tokens를 다시 CLS token으로
        self.tokens_to_noise = nn.Sequential(
            nn.Linear(self.hidden_size * num_noise_tokens, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, output_noise_dim)
        )
        
        # Learnable positional embeddings for noise tokens
        self.noise_pos_embeds = nn.Parameter(
            torch.randn(1, num_noise_tokens, self.hidden_size) * 0.02
        )

    def forward(self, x, t1, t2=None, y=None, noise_vector=None, args=None):
        if t2 is None:
            t2 = t1

        x_tokens = self.dit.x_embedder(x) + self.dit.pos_embed
        
        t1_embed = self.dit.t_embedder(t1)
        t2_embed = self.dit.t_embedder(t2)
        
        if y is not None:
            y_embed = self.dit.y_embedder(y, self.training)
            c = t1_embed + t2_embed + y_embed
        else:
            c = t1_embed + t2_embed
        
        # Noise vector를 multiple tokens으로 변환
        batch_size = x.size(0)
        noise_tokens = self.noise_to_tokens(noise_vector[0])  # [B, hidden_size * num_noise_tokens]
        noise_tokens = noise_tokens.view(batch_size, self.num_noise_tokens, self.hidden_size)
        noise_tokens = noise_tokens + self.noise_pos_embeds  # Positional encoding
        
        # 이미지 토큰과 결합
        all_tokens = torch.cat([x_tokens, noise_tokens], dim=1)
        
        features = []
        for i, block in enumerate(self.dit.blocks):
            all_tokens = block(all_tokens, c)
            if i in self.dit.encoder_depths:
                features.append(all_tokens.clone())
        
        num_patches = x_tokens.size(1)
        image_tokens = all_tokens[:, :num_patches, :]
        noise_tokens_out = all_tokens[:, num_patches:, :]  # [B, num_noise_tokens, hidden_size]
        
        image_output = self.dit.final_layer(image_tokens, c)
        image_output = self.dit.unpatchify(image_output)
        
        # Multiple tokens를 flatten하여 CLS token으로 변환
        noise_tokens_flat = noise_tokens_out.reshape(batch_size, -1)
        noise_output = self.tokens_to_noise(noise_tokens_flat)
        
        return image_output, noise_output, features



class DiTZeroflowintegrated_dino_generation(nn.Module):
    def __init__(self, original_dit, noise_dim=384, output_noise_dim=384, 
                 num_noise_tokens=16):  # CLS token을 16개 토큰으로 표현
        super().__init__()
        self.dit = original_dit
        self.num_noise_tokens = num_noise_tokens
        self.hidden_size = original_dit.pos_embed.size(-1)
        
        # Noise를 multiple tokens으로 변환
        self.noise_to_tokens = nn.Linear(noise_dim, self.hidden_size * num_noise_tokens)
        
        # Multiple tokens를 다시 CLS token으로
        self.tokens_to_noise = nn.Sequential(
            nn.Linear(self.hidden_size * num_noise_tokens, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, output_noise_dim)
        )
        
        # Learnable positional embeddings for noise tokens
        self.noise_pos_embeds = nn.Parameter(
            torch.randn(1, num_noise_tokens, self.hidden_size) * 0.02
        )

    def forward(self, x, t1, t2=None, y=None, noise_vector=None, args=None):
        if t2 is None:
            t2 = t1

        x_tokens = self.dit.x_embedder(x) + self.dit.pos_embed
        
        t1_embed = self.dit.t_embedder(t1)
        t2_embed = self.dit.t_embedder(t2)
        
        if y is not None:
            y_embed = self.dit.y_embedder(y, self.training)
            c = t1_embed + t2_embed + y_embed
        else:
            c = t1_embed + t2_embed
        
        # Noise vector를 multiple tokens으로 변환
        batch_size = x.size(0)
        noise_tokens = self.noise_to_tokens(noise_vector[0])  # [B, hidden_size * num_noise_tokens]
        noise_tokens = noise_tokens.view(batch_size, self.num_noise_tokens, self.hidden_size)
        noise_tokens = noise_tokens + self.noise_pos_embeds  # Positional encoding
        
        # 이미지 토큰과 결합
        all_tokens = torch.cat([x_tokens, noise_tokens], dim=1)
        
        features = []
        for i, block in enumerate(self.dit.blocks):
            all_tokens = block(all_tokens, c)
            if i in self.dit.encoder_depths:
                features.append(all_tokens.clone())
        
        num_patches = x_tokens.size(1)
        image_tokens = all_tokens[:, :num_patches, :]
        noise_tokens_out = all_tokens[:, num_patches:, :]  # [B, num_noise_tokens, hidden_size]
        
        image_output = self.dit.final_layer(image_tokens, c)
        image_output = self.dit.unpatchify(image_output)
        
        # Multiple tokens를 flatten하여 CLS token으로 변환
        noise_tokens_flat = noise_tokens_out.reshape(batch_size, -1)
        noise_output = self.tokens_to_noise(noise_tokens_flat)
        
        return image_output, noise_output, features