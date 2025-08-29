""" Group Query Attention """
"""
This class implements Optimized version of MultiHead Attention with Grouping Query Heads
Rotary Embedding to rotate entire query & key tensors
Creating causal mask by utilizing scaled_dot_product() from torch - SDPA (Huggingface style)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from finetunex.modules.positional_encoding import apply_rotary_emb

class GroupQueryAttention(nn.Module): #Qwen2 Attention
    def __init__(self, dim, num_head_q, num_head_kv, rope_theta, layer_idx = 0):
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

    def forward(self, x, position_emb,  attention_mask):
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
        q, k= apply_rotary_emb(q, k, cos, sin)

        # repeating kv heads to match query head
        k = k.repeat_interleave(self.num_head_q // self.num_head_kv, dim = 1)
        v = v.repeat_interleave(self.num_head_q // self.num_head_kv, dim = 1)
        attn_scores = q @ k.transpose(2, 3)  # (b, n_heads, q_len, k_len)
        attn_scores = attn_scores.masked_fill(attention_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores / self.headD **0.5, dim=-1)
        context = (attn_weights @ v).transpose(1, 2).reshape(B, T, self.headD * self.num_head_q)
        return self.o_proj(context)
    

class LlamaGroupQueryAttention(nn.Module): #Llama Attention
    def __init__(self, dim, num_head_q, num_head_kv, rope_theta, layer_idx = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.num_head_q = num_head_q
        self.num_head_kv = num_head_kv
        self.headD = dim // num_head_q
        assert self.headD * self.num_head_q == self.dim
        self.q_proj = nn.Linear(dim, self.headD * self.num_head_q, bias = False)
        self.k_proj = nn.Linear(dim, self.headD * self.num_head_kv, bias=False)
        self.v_proj = nn.Linear(dim, self.headD * self.num_head_kv, bias=False)
        self.o_proj = nn.Linear(self.headD * self.num_head_q, dim, bias=False)

    def forward(self, x, position_emb,  attention_mask):
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
        q, k= apply_rotary_emb(q, k, cos, sin)

        # repeating kv heads to match query head
        k = k.repeat_interleave(self.num_head_q // self.num_head_kv, dim = 1)
        v = v.repeat_interleave(self.num_head_q // self.num_head_kv, dim = 1)
        attn_scores = q @ k.transpose(2, 3)  # (b, n_heads, q_len, k_len)
        attn_scores = attn_scores.masked_fill(attention_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores / self.headD **0.5, dim=-1)
        context = (attn_weights @ v).transpose(1, 2).reshape(B, T, self.headD * self.num_head_q)
        return self.o_proj(context)