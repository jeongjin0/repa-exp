import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.vision_transformer import Attention
import torch.nn.functional as F
from einops import repeat, pack, unpack
from torch.cuda.amp import autocast




class DiTZeroflowintegrated_dino_generation(nn.Module):
    def __init__(self, original_dit, noise_dim=384, output_noise_dim=384, 
                 num_noise_tokens=16):  # CLS token을 16개 토큰으로 표현
        super().__init__()
        self.dit = original_dit
        self.num_noise_tokens = num_noise_tokens
        
        # s_embedder와 x_embedder의 출력 차원 가져오기
        self.s_hidden_size = original_dit.pos_embed.size(-1)  # 384
        self.x_hidden_size = original_dit.x_pos_embed.size(-1) if hasattr(original_dit, 'x_pos_embed') else self.s_hidden_size  # 2048 등
        
        # Encoder용 noise tokens (s_hidden_size 차원)
        self.noise_to_tokens_encoder = nn.Linear(noise_dim, self.s_hidden_size * num_noise_tokens)
        self.noise_pos_embeds_encoder = nn.Parameter(
            torch.randn(1, num_noise_tokens, self.s_hidden_size) * 0.02
        )
        
        # Decoder용 noise tokens (x_hidden_size 차원)
        self.noise_to_tokens_decoder = nn.Linear(noise_dim, self.x_hidden_size * num_noise_tokens)
        self.noise_pos_embeds_decoder = nn.Parameter(
            torch.randn(1, num_noise_tokens, self.x_hidden_size) * 0.02
        )
        
        # Output (decoder 차원 기준)
        self.tokens_to_noise = nn.Sequential(
            nn.Linear(self.x_hidden_size * num_noise_tokens, self.x_hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.x_hidden_size * 2, output_noise_dim)
        )


class DiTZeroflowintegrated_dino_generation(nn.Module):
    def __init__(self, original_dit, noise_dim=384, output_noise_dim=384, 
                 num_noise_tokens=16):
        super().__init__()
        self.dit = original_dit
        self.num_noise_tokens = num_noise_tokens
        
        # s_embedder와 x_embedder의 출력 차원 가져오기
        self.s_hidden_size = self.dit.encoder_hidden_size
        self.x_hidden_size = self.dit.decoder_hidden_size
        
        # Encoder용 noise tokens (s_hidden_size 차원)
        self.noise_to_tokens_encoder = nn.Linear(noise_dim, self.s_hidden_size * num_noise_tokens)
        self.noise_pos_embeds_encoder = nn.Parameter(
            torch.randn(1, num_noise_tokens, self.s_hidden_size) * 0.02
        )
        
        # Decoder용 noise tokens (x_hidden_size 차원)
        self.noise_to_tokens_decoder = nn.Linear(noise_dim, self.x_hidden_size * num_noise_tokens)
        self.noise_pos_embeds_decoder = nn.Parameter(
            torch.randn(1, num_noise_tokens, self.x_hidden_size) * 0.02
        )
        
        # Output (decoder 차원 기준)
        self.tokens_to_noise = nn.Sequential(
            nn.Linear(self.x_hidden_size * num_noise_tokens, self.x_hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.x_hidden_size * 2, output_noise_dim)
        )

    def forward(self, x, t1, t2=None, y=None, s=None, noise_vector=None, args=None):
        if t2 is None:
            t2 = t1

        # Time embeddings
        t1_embed = self.dit.t_embedder(t1)
        t2_embed = self.dit.t_embedder(t2)
        
        # Class embedding
        if y is not None:
            y_embed = self.dit.y_embedder(y, self.training)
            c = nn.functional.silu(t1_embed + t2_embed + y_embed)
        else:
            c = nn.functional.silu(t1_embed + t2_embed)

        batch_size = x.size(0)

        # ===== ENCODER 단계 =====
        # Encoder용 noise tokens (s_hidden_size 차원)
        noise_tokens_enc = self.noise_to_tokens_encoder(noise_vector[0])
        noise_tokens_enc = noise_tokens_enc.view(batch_size, self.num_noise_tokens, self.s_hidden_size)
        noise_tokens_enc = noise_tokens_enc + self.noise_pos_embeds_encoder

        # S embedding
        if s is None:
            s_tokens = self.dit.s_embedder(x)
            if self.dit.use_pos_embed and self.dit.pos_embed is not None:
                s_tokens = s_tokens + self.dit.pos_embed
            
            # Encoder에 이미지 + noise 결합 (같은 차원)
            s_all = torch.cat([s_tokens, noise_tokens_enc], dim=1)  # [B, num_patches + num_noise_tokens, s_hidden_size]
            
            # Encoder blocks
            for i in range(self.dit.num_encoder_blocks):
                s_all = self.dit.blocks[i](s_all, c)
            
            # Broadcast t to s
            t_broadcast = (t1_embed + t2_embed).unsqueeze(1).repeat(1, s_all.shape[1], 1)
            s_all = nn.functional.silu(t_broadcast + s_all)
            
            s = s_all
        
        # Project s
        s = self.dit.s_projector(s)  # s는 전체 (이미지 + noise)

        # ===== DECODER 단계 =====
        # Decoder용 noise tokens (x_hidden_size 차원)
        noise_tokens_dec = self.noise_to_tokens_decoder(noise_vector[0])
        noise_tokens_dec = noise_tokens_dec.view(batch_size, self.num_noise_tokens, self.x_hidden_size)
        noise_tokens_dec = noise_tokens_dec + self.noise_pos_embeds_decoder
        
        # X tokens for decoder
        x_tokens = self.dit.x_embedder(x)
        if self.dit.use_pos_embed and self.dit.x_pos_embed is not None:
            x_tokens = x_tokens + self.dit.x_pos_embed
        
        # Decoder에서 이미지 + noise 결합 (같은 차원)
        all_tokens = torch.cat([x_tokens, noise_tokens_dec], dim=1)  # [B, num_patches + num_noise_tokens, x_hidden_size]
        
        # Decoder blocks
        for i in range(self.dit.num_encoder_blocks, self.dit.num_blocks):
            all_tokens = self.dit.blocks[i](all_tokens, s)
        
        # 이미지 토큰과 noise 토큰 분리
        num_patches = x_tokens.size(1)
        image_tokens = all_tokens[:, :num_patches, :]
        noise_tokens_out = all_tokens[:, num_patches:, :]
        
        # Final outputs
        # s에서 이미지 부분만 사용
        s_image = s[:, :num_patches, :]
        image_output = self.dit.final_layer(image_tokens, s_image)
        image_output = self.dit.unpatchify(image_output)
        
        # Multiple tokens를 flatten하여 embedding으로 변환
        noise_tokens_flat = noise_tokens_out.reshape(batch_size, -1)
        noise_output = self.tokens_to_noise(noise_tokens_flat)
        
        return image_output, noise_output, None