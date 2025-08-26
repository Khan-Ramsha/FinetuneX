""" Base class: every model inherits base class"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._build_model()

    @abstractmethod
    def _build_model(self):
        """"Initializes layers, embeddings."""
        pass
    
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

    @abstractmethod
    def forward(self, input_ids, attention_mask = None):
        pass