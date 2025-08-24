""" Base class: every model inherits base class"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size) #Token embedding
        self.layers = nn.ModuleList(
            [self.decoder_layer(self.config, layer_idx) for layer_idx in range(self.config.num_hidden_layers)]
        )
        self.norm = self.norm_layer(config.hidden_size, config.rms_norm_eps)
        self.rotary = self.rotary_embedding(self.config.hidden_size // self.config.num_attention_heads, config.rope_theta)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)        
        
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight #tie embeddings

    @abstractmethod
    def decoder_layer(self, layer_idx) -> nn.Module:
        """ Model specific decoder layer implementation"""
        pass

    @abstractmethod
    def norm_layer(self, hidden_size, eps) -> nn.Module:
        """Normalization Layer for model"""
        pass
    
    @abstractmethod
    def rotary_embedding(self, dim, rope_theta) -> nn.Module:
        """" Positional Encoding"""
        pass

    # @abstractmethod
    # def attention_layer(self) -> nn.Module:
    #     """"Model-specific Attention implementation"""
    #     pass
    
    # @abstractmethod
    # def feedforward(self) -> nn.Module:
    #     """"MLP implementation"""
    #     pass
    
    @abstractmethod
    def forward(self, x, attention_mask = None):
        pass