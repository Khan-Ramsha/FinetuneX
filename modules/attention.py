"""TODO: Group Query Attention """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.headD = dim // num_heads
        assert self.headD * self.num_heads == self.dim
        self.to_q = nn.Linear(dim, dim, bias = True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim,bias=True)
        self.output_proj = nn.Linear(dim, dim,bias=True)
    
    def forward(self, x):
        B, T, D = x.shape
        q = self.to_q(x) #linear transformation, x@Wq + b (Wq learnable weight for query)
        k = self.to_k(x)
        v = self.to_v(x)
        # q, k, v each of [B, T, D] but since multiple heads D is to be split num_heads * head_dim
        #reshape & transpose results => [B, H, T, Hd]
        q = q.view(B, -1, self.num_heads, self.headD).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.headD).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.headD).transpose(1, 2)
        #scaled dot-product
        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.headD)
        scores = F.softmax(att_scores, dim = -1)
        att_scores = scores @ v
        output = att_scores.transpose(1, 2).contiguous().view(B, T, D)
        output = self.output_proj(output)
        return output

    