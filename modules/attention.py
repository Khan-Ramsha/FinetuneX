"""TODO: Group Query Attention """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):
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
    
    def forward(self, x, past_kv = None):
        B, T, D = x.shape
        q = self.to_q(x) #linear transformation, x@Wq + b (Wq learnable weight for query)
        k = self.to_k(x)
        v = self.to_v(x)
        # q, k, v each of [B, T, D] but since multiple heads D is to be split num_head_q * head_dim
        #reshape  => [B, T, H, Hd] & transpose [B,T,H, Hd] => [B, H, T, Hd]
        q = q.view(B, -1, self.num_head_q, self.headD).transpose(1,2)
        k = k.view(B, -1, self.num_head_kv, self.headD).transpose(1,2)
        v = v.view(B, -1, self.num_head_kv, self.headD).transpose(1,2)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            #Concatenate 
            k = torch.cat((past_k, k), dim = 2)
            v = torch.cat((past_v, v), dim = 2)
        present_kv = (k,v)
        # repeating kv heads to match query head
        k = k.repeat_interleave(self.num_head_q // self.num_head_kv, dim = 1)
        v = v.repeat_interleave(self.num_head_q // self.num_head_kv, dim = 1)
        #scaled dot-product
        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.headD)
        q_len = q.size(-2)
        k_len = k.size(-2)
        mask = torch.tril(torch.ones(q_len, k_len),device=att_scores.device, dtype=att_scores.dtype).unsqueeze(0).unsqueeze(0)
        att_weights = att_scores.masked_fill_(mask == 0, float("-inf"))
        att_probs = F.softmax(att_weights, dim = -1)
        output = att_probs @ v
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.output_proj(output) #linear transformation to extract useful info from output
        return output, present_kv # present_kv will be the past_kv for next steps (during inference only)