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
from positional_encoding import RotaryEmbedding

class GroupQueryAttention(nn.Module):
    def __init__(self, dim, num_head_q, num_head_kv):
        super().__init__()
        self.dim = dim
        self.num_head_q = num_head_q
        self.num_head_kv = num_head_kv
        self.headD = dim // num_head_q
        assert self.headD * self.num_head_q == self.dim
        self.to_q = nn.Linear(dim, self.headD * self.num_head_q, bias = True)
        self.to_k = nn.Linear(dim, self.headD * self.num_head_kv, bias=True)
        self.to_v = nn.Linear(dim, self.headD * self.num_head_kv, bias=True)
        self.output_proj = nn.Linear(self.headD * self.num_head_q, dim, bias=False)
        self.rotary = RotaryEmbedding(self.headD)

    def forward(self, x, past_kv = None, attention_mask = None):
        B, T, D = x.shape
        q = self.to_q(x) #linear transformation, x@Wq + b (Wq learnable weight for query)
        k = self.to_k(x)
        v = self.to_v(x)
        # q, k, v each of [B, T, D] but since multiple heads D is to be split num_head_q * head_dim
        #reshape  => [B, T, H, Hd] & transpose [B,T,H, Hd] => [B, H, T, Hd]
        q = q.view(B, -1, self.num_head_q, self.headD).transpose(1,2)
        k = k.view(B, -1, self.num_head_kv, self.headD).transpose(1,2)
        v = v.view(B, -1, self.num_head_kv, self.headD).transpose(1,2)
        
        #rotate key, query
        q = self.rotary(q)
        k = self.rotary(k)

        #kv caching
        if past_kv is not None:
            past_k, past_v = past_kv
            #Concatenate 
            k = torch.cat((past_k, k), dim = 2)
            v = torch.cat((past_v, v), dim = 2)
        present_kv = (k,v)
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
        output = self.output_proj(output) #linear transformation to extract useful info from output 
        return output, present_kv # present_kv will be the past_kv for next steps (during inference only)