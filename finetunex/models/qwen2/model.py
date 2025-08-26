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
from torch.utils.checkpoint import checkpoint

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

    def _build_model(self):
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size) #Token embedding
        self.layers = nn.ModuleList(
            [self.decoder_layer(self.config, layer_idx) for layer_idx in range(self.config.num_hidden_layers)]
        )
        self.norm = self.norm_layer(self.config.hidden_size, self.config.rms_norm_eps)
        self.rotary = self.rotary_embedding(self.config.hidden_size // self.config.num_attention_heads, self.config.rope_theta)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)        
        
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight #tie embeddings
    
    def rotary_embedding(self, dim, base):
        return RotaryEmbedding(dim, base)
    
    def norm_layer(self, hidden_size, eps):
        return RMSNorm(hidden_size, eps)

    def decoder_layer(self, config, layer_idx):
        return DecoderBlock(config, layer_idx)
    
    def forward(self, input_ids, attention_mask = None, labels = None):
        B, T = input_ids.shape
        hidden_states = self.embed_tokens(input_ids) 
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.rotary(hidden_states, position_ids)
        # passing hidden states to stack of blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states,pos_emb, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        print(f"BEFORE LOSS CALCULATION:")
        print(f"logits.shape: {logits.shape}")
        print(f"logits.dim(): {logits.dim()}")
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            try:
                shift_logits = logits[..., :-1, :].contiguous()
            except Exception as e:
                raise e
                
            try:
                shift_labels = labels[..., 1:].contiguous()
            except Exception as e:
                raise e
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            print(f"Calculated loss: {loss}")                         
        return {
            'loss': loss,
            'logits': logits
        }