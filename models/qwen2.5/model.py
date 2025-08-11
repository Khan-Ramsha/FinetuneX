"""Transformer Block & main architecture"""
import torch
import torch.nn as nn
from ...modules.attention import GroupQueryAttention
from ...modules.norm import RMSNorm
from ...modules.mlp import MLP
from ...base.config import Config
from ...modules.positional_encoding import RotaryEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, config: Config, i : int):
        super().__init__()
        self.pre_ln = RMSNorm(config.hidden_size)
        self.attn = GroupQueryAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads)
        self.post_ln = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)
    
    def forward(self, x, past_kv = None):
        x = x + self.attn(self.pre_ln(x))
        x = x + self.mlp(self.post_ln(x))
        return x

class Qwen2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size) #Token embedding
        self.layers = nn.ModuleList(
            [DecoderBlock(self.config, i) for i in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        wte = self.embed_tokens(x) 
        # a passing input to stack of blocks
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits