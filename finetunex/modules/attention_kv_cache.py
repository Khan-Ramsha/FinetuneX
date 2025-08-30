import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""there is no casual masking in the code. For decoder, add a proper casual masking"""
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
    
    def forward(self, x,  past_kv = None):
        B, T, D = x.shape
        q = self.to_q(x) #linear transformation, x@Wq + b (Wq learnable weight for query)
        k = self.to_k(x)
        v = self.to_v(x)
        # q, k, v each of [B, T, D] but since multiple heads D is to be split num_heads * head_dim
        #reshape & transpose results => [B, H, T, Hd]
        q = q.view(B, -1, self.num_heads, self.headD).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.headD).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.headD).transpose(1, 2)
        
        present_kv = (k,v)
        if past_kv is not None:
            past_k, past_v = past_kv
            #Concatenate 
            k = torch.cat((past_k, k), dim = 2)
            v = torch.cat((past_v, v), dim = 2)
            present_kv = (k,v)
        #scaled dot-product
        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.headD)
        att_probs = F.softmax(att_scores, dim = -1)
        att_scores = att_probs @ v
        output = att_scores.transpose(1, 2).contiguous().view(B, T, D)
        output = self.output_proj(output)
        return output, present_kv