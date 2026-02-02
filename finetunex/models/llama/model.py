"""Transformer Block & main architecture"""
import torch
import torch.nn as nn
from finetunex.base.model import BaseModel
from finetunex.base.config import Config
from finetunex.modules.attention import LlamaGroupQueryAttention
from finetunex.modules.norm import RMSNorm
from finetunex.modules.mlp import MLP
from finetunex.modules.positional_encoding import RotaryEmbedding

class LlamaDecoderBlock(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = LlamaGroupQueryAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.rope_theta, layer_idx = layer_idx, use_flashattn = False)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)
    
    def forward(self, x, pos_emb, attention_mask):
        residuals = x
        x = self.input_layernorm(x)
        x = self.attn(x, pos_emb, attention_mask)
        x = residuals + x
        residuals = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residuals + x
        return x

class LlamaModel(BaseModel):
    def _build_model(self):
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size) 
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
    
    def rotary_embedding(self, dim, base):
        return RotaryEmbedding(dim, base)
    
    def norm_layer(self, hidden_size, eps):
        return RMSNorm(hidden_size, eps)

    def decoder_layer(self, config, layer_idx):
        return LlamaDecoderBlock(config, layer_idx)
    
    def forward(self, input_ids, attn_mask = None,labels = None):
        B, T = input_ids.shape
        input_embeds = self.embed_tokens(input_ids) 
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        hidden_states = input_embeds
        pos_emb = self.rotary(hidden_states, position_ids)
        
        num_tokens = hidden_states.shape[1]
        causal_mask = torch.triu(torch.ones(num_tokens, num_tokens, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        if attn_mask is not None:
            padding_mask = (attn_mask == 0) #HIDE for value zero 
            combined_mask = causal_mask[None, None, :, :] | padding_mask[:, None, None, :]
        else:
            combined_mask = causal_mask[None, None, :, :] #broadcasting
        # passing hidden states to stack of blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_emb, combined_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
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
        return {
            'loss': loss,
            'logits': logits
        }