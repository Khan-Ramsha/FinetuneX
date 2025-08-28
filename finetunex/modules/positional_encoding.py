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
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="cuda") / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids = None):
        *p, seq_len, dim = x.shape
        inv_freq = self.inv_freq
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        if position_ids is not None:
            # Use provided position_ids
            position_ids_expanded = position_ids[:, None, :].float()
        else:
            position_ids_expanded = torch.arange(seq_len, device=x.device).type_as(inv_freq) # positions
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
"""rotating key values"""
def rotate_half(x):
    #split into first and second half
    x1 = x[..., : x.shape[-1] // 2] 
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1) #shape: [batch_size, :, seq_len, head_dim] making compatible to q & v shape
    sin = sin.unsqueeze(1)
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb