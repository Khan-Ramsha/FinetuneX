"""Base class: every model inherits base class"""
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config, args=None):
        super().__init__()
        self.config = config
        self.args = args
        self._build_model()

    def _build_model(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_model()")

    def decoder_layer(self, config, layer_idx) -> nn.Module:
        raise NotImplementedError(f"{self.__class__.__name__} must implement decoder_layer()")

    def norm_layer(self, hidden_size, eps) -> nn.Module:
        raise NotImplementedError(f"{self.__class__.__name__} must implement norm_layer()")

    def rotary_embedding(self, dim, base) -> nn.Module:
        raise NotImplementedError(f"{self.__class__.__name__} must implement rotary_embedding()")

    def forward(self, input_ids, attention_mask=None, labels=None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")