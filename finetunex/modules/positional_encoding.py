"""RoPE(Rotary Position Encoding)"""
import torch
import torch.nn as nn
"""
    Core Idea
    - For each position m, rotate embedding by angle m*θ
    - θ_i = 1 / (10000^(2i/d)) for dimension pair i
    - Apply 2D rotation to each pair of dimensions
"""
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_theta): 
        super().__init__()
        self.dim = dim
        self.rope_theta = rope_theta # base frequency
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids = None):
        *p, seq_len, dim = x.shape
        inv_freq = self.inv_freq
        if position_ids is not None:
            # Use provided position_ids
            if position_ids.dim() > 1:
                t = position_ids[0].float().type_as(inv_freq) 
            else: 
                t = position_ids.float().type_as(inv_freq) 
        else:
            t = torch.arange(seq_len, device=x.device).type_as(inv_freq) # positions
        freqs = torch.einsum('i,j->ij', t, inv_freq) # t @ inv_freq (matmul)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        cos = emb.cos()
        sin = emb.sin()
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :] # to match [B, H, seq_len, head_dim]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
"""rotating key values"""
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_emb(q, k, cos, sin):
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb