"""Transformer Block & main architecture"""
import torch
import torch.nn as nn
from finetunex.base.model import BaseModel
from finetunex.models.qwen2 import modell
from finetunex.base.config import Config
from finetunex.modules.attention import GroupQueryAttention
from finetunex.modules.norm import RMSNorm
from finetunex.modules.mlp import MLP
from finetunex.modules.positional_encoding import RotaryEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, config: Config, layer_idx : int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = GroupQueryAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.rope_theta, layer_idx = layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)
    
    def forward(self, x,pos_emb=None, attention_mask = None):
        x = x + self.attn(self.input_layernorm(x), pos_emb, attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Qwen2Model(BaseModel):
    
    def rotary_embedding(self, dim, base):
        return RotaryEmbedding(dim, base)
    
    def norm_layer(self, hidden_size, eps):
        return RMSNorm(hidden_size, eps)

    def decoder_layer(self, config, layer_idx):
        return DecoderBlock(config, layer_idx)
    
    def forward(self, x, attention_mask = None):
        B, T = x.shape
        hidden_states = self.embed_tokens(x) 
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.rotary(hidden_states, position_ids)
        # passing hidden states to stack of blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states,pos_emb, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits