import math
from functools import lru_cache
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .configuration_custom_dit import CustomDiTConfig


# class RotaryPositionEmbedding(nn.Module):
#     def __init__(self, hidden_size, num_heads, rope_theta=10000):
#         super().__init__()
#         assert hidden_size % num_heads == 0
        
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads

#         self.rope_theta = rope_theta

#         self.thetas = torch.pow(rope_theta, -2 * torch.arange(self.head_dim // 2) / self.head_dim)

#         # [1024, H/2]
#         self.sines = torch.sin(self.thetas.unsqueeze(0) * torch.arange(1024).unsqueeze(1))
#         self.cosines = torch.cos(self.thetas.unsqueeze(0) * torch.arange(1024).unsqueeze(1))


#     def forward(self, query, key, query_positions=None, key_positions=None):
#         _, _, q_seq_len, _ = query.shape
#         _, _, k_seq_len, _ = key.shape
#         device = query.device

#         if k_seq_len > self.sines.shape[0] or self.sines.is_meta:
#             self.thetas = torch.pow(self.rope_theta, -2 * torch.arange(self.head_dim // 2) / self.head_dim)

#             self.sines = torch.sin(self.thetas.unsqueeze(0) * torch.arange(k_seq_len).unsqueeze(1))
#             self.cosines = torch.cos(self.thetas.unsqueeze(0) * torch.arange(k_seq_len).unsqueeze(1))

#         if self.sines.device != device:
#             self.sines = self.sines.to(device)
#             self.cosines = self.cosines.to(device)

#         query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
#         key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

#         query_real_rotated = torch.zeros_like(query, device=device)
#         query_imag_rotated = torch.zeros_like(query, device=device)
#         key_real_rotated = torch.zeros_like(key, device=device)
#         key_imag_rotated = torch.zeros_like(key, device=device)

#         query_real_rotated[..., 0::2] = query_real * self.cosines[:q_seq_len]
#         query_real_rotated[..., 1::2] = query_real * self.sines[:q_seq_len]

#         query_imag_rotated[..., 0::2] = -query_imag * self.sines[:q_seq_len]
#         query_imag_rotated[..., 1::2] = query_imag * self.cosines[:q_seq_len]

#         key_real_rotated[..., 0::2] = key_real * self.cosines[:k_seq_len]
#         key_real_rotated[..., 1::2] = key_real * self.sines[:k_seq_len]

#         key_imag_rotated[..., 0::2] = -key_imag * self.sines[:k_seq_len]
#         key_imag_rotated[..., 1::2] = key_imag * self.cosines[:k_seq_len]

#         query_out = query_real_rotated + query_imag_rotated
#         key_out = key_real_rotated + key_imag_rotated

#         return query_out, key_out


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_theta=10000):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.rope_theta = rope_theta

        thetas = torch.pow(rope_theta, -2 * torch.arange(self.head_dim // 2) / self.head_dim)

        # [1024, H/2]
        # self.sines = torch.sin(thetas.unsqueeze(0) * torch.arange(1024).unsqueeze(1))
        # self.cosines = torch.cos(thetas.unsqueeze(0) * torch.arange(1024).unsqueeze(1))

        self.register_buffer('sines', torch.sin(thetas.unsqueeze(0) * torch.arange(1024).unsqueeze(1)), persistent=False)
        self.register_buffer('cosines', torch.cos(thetas.unsqueeze(0) * torch.arange(1024).unsqueeze(1)), persistent=False)


    def forward(self, query, key, query_positions=None, key_positions=None):
        B, _, q_seq_len, _ = query.shape
        _, _, k_seq_len, _ = key.shape
        device = query.device

        if query_positions is None:
            query_positions = torch.arange(q_seq_len).unsqueeze(0).repeat(B, 1)

        if key_positions is None:
            key_positions = torch.arange(k_seq_len).unsqueeze(0).repeat(B, 1)

        max_pos = max(key_positions.max().item(), query_positions.max().item())

        if max_pos >= self.sines.shape[0] or self.sines.is_meta:
            thetas = torch.pow(self.rope_theta, -2 * torch.arange(self.head_dim // 2) / self.head_dim)

            self.sines = torch.sin(thetas.unsqueeze(0) * torch.arange(max_pos+1).unsqueeze(1)).to(device)
            self.cosines = torch.cos(thetas.unsqueeze(0) * torch.arange(max_pos+1).unsqueeze(1)).to(device)

        query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
        key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

        query_real_rotated = torch.zeros_like(query, device=device)
        query_imag_rotated = torch.zeros_like(query, device=device)
        key_real_rotated = torch.zeros_like(key, device=device)
        key_imag_rotated = torch.zeros_like(key, device=device)

        query_real_rotated[..., 0::2] = query_real * self.cosines[query_positions].unsqueeze(1)
        query_real_rotated[..., 1::2] = query_real * self.sines[query_positions].unsqueeze(1)

        query_imag_rotated[..., 0::2] = -query_imag * self.sines[query_positions].unsqueeze(1)
        query_imag_rotated[..., 1::2] = query_imag * self.cosines[query_positions].unsqueeze(1)

        key_real_rotated[..., 0::2] = key_real * self.cosines[key_positions].unsqueeze(1)
        key_real_rotated[..., 1::2] = key_real * self.sines[key_positions].unsqueeze(1)

        key_imag_rotated[..., 0::2] = -key_imag * self.sines[key_positions].unsqueeze(1)
        key_imag_rotated[..., 1::2] = key_imag * self.cosines[key_positions].unsqueeze(1)

        query_out = query_real_rotated + query_imag_rotated
        key_out = key_real_rotated + key_imag_rotated

        assert torch.isfinite(self.sines).all(), "RoPE sines contain NaN/Inf"
        assert torch.isfinite(self.cosines).all(), "RoPE cosines contain NaN/Inf"

        return query_out, key_out


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 bias=True,
                 rope_theta=10000.0,
                 attn_dropout=0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                '`hidden_size`({}) should be divisible by `num_heads`({})'.format(hidden_size, num_heads))
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.bias = bias
        self.linear_q = nn.Linear(hidden_size, hidden_size, bias)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias)
        self.linear_o = nn.Linear(hidden_size, hidden_size, bias)

        self.rope = RotaryPositionEmbedding(hidden_size, num_heads, rope_theta)

        self.batch_first = True
        self.head_dim = hidden_size // num_heads

        self.attn_dropout = attn_dropout

    def forward(self, q, k, v, attention_mask=None, q_positions=None, k_positions=None):
        # B, L, E
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        q = self._reshape_to_mha(q)
        k = self._reshape_to_mha(k)
        v = self._reshape_to_mha(v)

        # B, H, L, e
        q, k = self.rope(q, k, q_positions, k_positions)

        y = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=self.attn_dropout if self.training else 0.0)
        # attn_weight = q @ k.transpose(-2, -1)
        # attn_weight = torch.softmax(attn_weight, dim=-1)
        # if self.training:
        #     attn_weight = F.dropout(attn_weight, self.attn_dropout)
        # y = attn_weight @ v

        y = self._reshape_from_mha(y)
        # B, L, E
        y = self.linear_o(y)

        return y

    def _reshape_to_mha(self, x):
        batch_size, seq_len, hidden_size = x.size()
        sub_dim = hidden_size // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3)

    def _reshape_from_mha(self, x):
        batch_size, num_heads, seq_len, sub_dim = x.size()
        hidden_size = sub_dim * num_heads
        return x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)


class AdaLN(nn.Module):
    def __init__(self, hidden_dim, act=nn.SiLU()):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, x, emb):
        gamma, beta = self.mlp(emb).chunk(2, dim=-1)

        return (gamma + 1) * x + beta


class FourierTimestepEmbedding(nn.Module):
    def __init__(self, num_freqs=32, f_max=100):
        super().__init__()

        self.num_freqs = num_freqs
        freqs = torch.as_tensor([2 ** (i * math.log2(f_max) / (num_freqs - 1)) for i in range(num_freqs)])

        self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, t):
        # t: [B, L], freqs: [1, F]
        rot = (2 * math.pi * t).unsqueeze(-1) * self.freqs
        sines = torch.sin(rot)
        cosines = torch.cos(rot)

        return torch.stack([sines, cosines], dim=-1).view(t.shape[0], t.shape[1], -1)


class ContinuationAttnDiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, rope_theta=500000, layer_norm_eps=1e-6, 
        mlp_act=nn.SiLU(), dropout=0.05, adaLN_act=nn.SiLU()):
        super().__init__()

        self.pre_attn_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.pre_attn_adaLN = AdaLN(hidden_dim=hidden_dim, act=adaLN_act)
        self.attn = MultiHeadAttentionWithRoPE(hidden_dim, num_heads, rope_theta=rope_theta, attn_dropout=dropout)

        self.pre_mlp_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.pre_mlp_adaLN = AdaLN(hidden_dim=hidden_dim, act=adaLN_act)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            mlp_act,
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, context, timestep_emb, attention_mask):
        B = x.shape[0]

        hidden = self.pre_attn_norm(x)
        hidden = self.pre_attn_adaLN(hidden, timestep_emb)

        context_with_hidden = torch.cat([context, hidden], dim=1)

        # TODO: fix
        assert attention_mask is None

        k_positions = torch.arange(context_with_hidden.shape[1]).to(x.device).unsqueeze(0).repeat(B, 1)
        q_positions = torch.arange(context.shape[1], context_with_hidden.shape[1]).to(x.device).unsqueeze(0).repeat(B, 1)

        hidden = self.attn(hidden, context_with_hidden, context_with_hidden, attention_mask, q_positions, k_positions)

        # context_mass = attn_scores[..., :128].sum(dim=-1).mean().item()
        # canvas_mass = attn_scores[..., 128:].sum(dim=-1).mean().item()

        # print(self.block_num, context_mass, canvas_mass, canvas_mass / context_mass, (hidden.norm() / x.norm()).item())

        hidden = hidden + x

        hidden2 = self.pre_mlp_norm(hidden)
        hidden2 = self.pre_mlp_adaLN(hidden2, timestep_emb)
        hidden2 = self.ffn(hidden2)

        hidden2 = hidden2 + hidden

        return hidden2


class CustomDiTModel(PreTrainedModel):
    config_class = CustomDiTConfig

    def __init__(self, config: CustomDiTConfig):
        super().__init__(config)

        if config.act == 'silu':
            self.act = nn.SiLU()
        elif config.act == 'relu':
            self.act = nn.ReLU()
        elif config.act == 'gelu':
            self.act = nn.GeLU('tanh')
        else:
            raise ValueError(f'unsupported activation fn {config.act}')

        if config.adaLN_act == 'silu':
            self.adaLN_act = nn.SiLU()
        elif config.adaLN_act == 'relu':
            self.adaLN_act = nn.ReLU()
        elif config.adaLN_act == 'gelu':
            self.adaLN_act = nn.GeLU('tanh')
        else:
            raise ValueError(f'unsupported activation fn {config.adaLN_act}')

        self.num_heads = config.num_heads

        if config.txt_dim != config.hidden_dim:
            self.input_proj = nn.Linear(config.txt_dim, config.hidden_dim)
        else:
            self.input_proj = nn.Identity()

        self.timestep_emb = FourierTimestepEmbedding(config.time_emb_num_freqs, config.time_emb_f_max)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(config.time_emb_num_freqs * 2, config.hidden_dim),
            self.act,
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        self.blocks = nn.ModuleList([
            ContinuationAttnDiTBlock(
                config.hidden_dim, 
                config.num_heads, 
                rope_theta=config.rope_theta, 
                layer_norm_eps=config.layer_norm_eps,
                mlp_act=self.act,
                dropout=config.attn_dropout,
                adaLN_act=self.adaLN_act
            ) for _ in range(config.num_layers)
        ])

        if config.txt_dim != config.hidden_dim:
            self.output_proj = nn.Linear(config.hidden_dim, config.txt_dim)
        else:
            self.output_proj = nn.Identity()

        self.post_init()

    def forward(self, txt_latent, context_latent, timestep, attention_mask):
        timestep_emb = self.timestep_emb(timestep)
        timestep_emb = self.timestep_mlp(timestep_emb)

        txt = self.input_proj(txt_latent)
        context = self.input_proj(context_latent)

        for i, block in enumerate(self.blocks):
            txt = block(txt, context, timestep_emb, attention_mask=None)

        txt = self.output_proj(txt)

        return txt


AutoConfig.register("custom_dit", CustomDiTConfig)
AutoModel.register(CustomDiTConfig, CustomDiTModel)
