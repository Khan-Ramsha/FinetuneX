""" Group Query Attention """
"""
This class implements Optimized version of MultiHead Attention with Grouping Query Heads using KV caching when in inference mode
Rotary Embedding to rotate entire query & key tensors
Creating causal mask by utilizing scaled_dot_product() from torch - SDPA (Huggingface style)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from finetunex.modules.positional_encoding import RotaryEmbedding

class GroupQueryAttention(nn.Module):
    def __init__(self, dim, num_head_q, num_head_kv, layer_idx = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.num_head_q = num_head_q
        self.num_head_kv = num_head_kv
        self.headD = dim // num_head_q
        assert self.headD * self.num_head_q == self.dim
        self.q_proj = nn.Linear(dim, self.headD * self.num_head_q, bias = True)
        self.k_proj = nn.Linear(dim, self.headD * self.num_head_kv, bias=True)
        self.v_proj = nn.Linear(dim, self.headD * self.num_head_kv, bias=True)
        self.o_proj = nn.Linear(self.headD * self.num_head_q, dim, bias=False)
        self.rotary = RotaryEmbedding(self.headD)

    def forward(self, x, position_emb,  attention_mask = None):
        B, T, D = x.shape
        q = self.q_proj(x) #linear transformation, x@Wq + b (Wq learnable weight for query)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # q, k, v each of [B, T, D] but since multiple heads D is to be split num_head_q * head_dim
        #reshape  => [B, T, H, Hd] & transpose [B,T,H, Hd] => [B, H, T, Hd]
        q = q.view(B, -1, self.num_head_q, self.headD).transpose(1,2)
        k = k.view(B, -1, self.num_head_kv, self.headD).transpose(1,2)
        v = v.view(B, -1, self.num_head_kv, self.headD).transpose(1,2)
        
        #rotate key, query
        cos, sin = position_emb
        q, k= self.rotary.apply_rotary_emb(q, k, cos, sin)

        # repeating kv heads to match query head
        k = k.repeat_interleave(self.num_head_q // self.num_head_kv, dim = 1)
        v = v.repeat_interleave(self.num_head_q // self.num_head_kv, dim = 1)

        # causal masking
        causal_mask = attention_mask
        if attention_mask is not None: #slicing
            causal_mask = attention_mask[:, :, :, : k.shape[-2]]
        #handling non-contiguity of tensors on cuda
        if q.device.type == "cuda" and attention_mask is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        # if attention mask not provided & not during inference, put causal = True
        is_causal = True if causal_mask is None and T > 1 else False
        att_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_mask,
            is_causal=is_causal
        )
        output = att_output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.o_proj(output) #linear transformation to extract useful info from output 
        return output