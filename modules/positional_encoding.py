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
    def __init__(self, dim, base = 10000): #TODO: base value to be used from model config "rope_theta"
        super().__init__()
        self.dim = dim
        self.base = base # base frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        print(f"Inverse Frequency: {inv_freq}")
    
    def rotate_half(self,x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim = -1)

    def apply_rotary_emb(self, q, k, cos, sin):
        q_emb = (q * cos) + (self.rotate_half(q) * sin)
        k_emb = (k * cos) + (self.rotate_half(k) * sin)
        return q_emb, k_emb
    
    def forward(self, x, position_ids = None):
        *p, seq_len, dim = x.shape
        if position_ids is not None:
            # Use provided position_ids
            seq_len = position_ids.shape[-1]
            t = position_ids.float().type_as(self.inv_freq)
        else:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq) # positions
        freqs = torch.einsum('i,j->ij', t, self.inv_freq) # t @ inv_freq (matmul)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        cos = emb.cos()
        sin = emb.sin()
        cos = cos[None, None, :, :]
        cos = cos[None, None, :, :] # to match [B, H, seq_len, head_dim]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)