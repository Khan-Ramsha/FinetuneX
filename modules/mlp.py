"""SwiGLU (activation function) in Gated MLP"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size 
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.silu = nn.SiLU()
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)

    def forward(self, x):
        x = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))
        return x