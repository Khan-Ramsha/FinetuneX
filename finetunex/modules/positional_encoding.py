"""RoPE(Rotary Position Encoding)"""
import torch
import torch.nn as nn
import math
"""
    Core Idea
    - For each position m, rotate embedding by angle m*θ
    - θ_i = 1 / (10000^(2i/d)) for dimension pair i
    - Apply 2D rotation to each pair of dimensions
"""

# Rotary embedding with rope scaling for LLama3
def _compute_llama3_inv_freq(dim, base, factor, low_freq_factor, high_freq_factor, old_context_len):
    """Exactly matches HuggingFace's Llama-3 RoPE scaling."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    low_freq_wavelen  = old_context_len / low_freq_factor   # 8192 / 1.0 = 8192
    high_freq_wavelen = old_context_len / high_freq_factor  # 8192 / 4.0 = 2048
    wavelen = 2 * math.pi / inv_freq

    # smooth interpolation factor (only used in medium band)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )

    # Interpolate between scaled (low-freq) and original (high-freq)
    smoothed = (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq

    new_inv_freq = torch.where(
        wavelen < high_freq_wavelen,
        inv_freq,               # high freq → keep original
        torch.where(
            wavelen > low_freq_wavelen,
            inv_freq / factor,  # low freq  → scale down
            smoothed,           # medium    → interpolate
        ),
    )
    return new_inv_freq

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, rope_theta: float, rope_scaling: dict = None):
        super().__init__()
        self.dim = dim
        self.rope_theta = rope_theta

        if rope_scaling is None or rope_scaling.get("rope_type", "default") == "default":
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        elif rope_scaling["rope_type"] == "llama3":
            inv_freq = _compute_llama3_inv_freq(
                dim=dim,
                base=rope_theta,
                factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                old_context_len=rope_scaling["original_max_position_embeddings"],
            )
        else:
            raise NotImplementedError(f"rope_type={rope_scaling['rope_type']!r} not supported")

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids=None):
        B = x.shape[0]
        T = x.shape[-2]
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(B, -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = "cpu" if x.device.type == "mps" else x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1) #shape: [batch_size, :, seq_len, head_dim] making compatible to q & v shape
    sin = sin.unsqueeze(1)
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb