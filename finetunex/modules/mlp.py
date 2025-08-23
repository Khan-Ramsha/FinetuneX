"""SwiGLU (activation function) in Gated MLP"""
import torch
import torch.nn as nn
from finetunex.base.config import Config

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))
        return x