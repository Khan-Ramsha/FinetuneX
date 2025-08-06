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
    def __init__(self, dim, base = 10000):
        super().__init__()
        self.dim = dim
        self.base = base # base frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        print(f"Inverse Frequency: {inv_freq}")
    
    def apply_rotation(self, x, cos, sin):
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        rotated_x1 = x_even * cos - x_odd * sin
        rotated_x2 = x_even * sin + x_odd * cos
        # Interleave (putting back in o rder)
        x_rot = torch.zeros_like(x)
        x_rot[..., ::2] = rotated_x1
        x_rot[..., 1::2] = rotated_x2
        return x_rot
    
    def forward(self, x):
        *p, seq_len, dim = x.shape
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq) # positions
        freqs = torch.einsum('i,j->ij', t, self.inv_freq) # t @ inv_freq (matmul)
        cos = freqs.cos()[None, :, :]
        sin = freqs.sin()[None, :, :]
        return self.apply_rotation(x,cos,sin)