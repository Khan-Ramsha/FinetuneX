"""Transformer Block & main architecture"""
import torch
import torch.nn as nn
from ...modules.attention import GroupQueryAttention
from ...modules.norm import RMSNorm
from ...modules.mlp import MLP
from ...base.config import Config
from ...modules.positional_encoding import RotaryEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, config: Config, layer_idx : int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.attn = GroupQueryAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, layer_idx = layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)
    
    def forward(self, x, past_kv = None):
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Qwen2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size) #Token embedding
        self.layers = nn.ModuleList(
            [DecoderBlock(self.config, layer_idx) for layer_idx in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size)
        self.rotary = RotaryEmbedding(self.config.hidden_size // self.config.num_attention_heads)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight #tie embeddings
        
    def forward(self, x, attention_mask = None):
        B, T = x.shape
        hidden_states = self.embed_tokens(x) 
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.rotary(hidden_states, position_ids)
        # passing hidden states to stack of blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_emb, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits