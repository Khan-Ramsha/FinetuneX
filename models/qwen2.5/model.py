"""Transformer Block & main architecture"""
import torch.nn as nn
from ...modules.attention import GroupQueryAttention
from ...modules.norm import RMSNorm
from ...modules.mlp import MLP

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_ln = RMSNorm()
        self.attn = GroupQueryAttention()
        self.post_ln = RMSNorm()
        self.mlp = MLP()
    
    def forward(self, x):
        x = x + self.attn(self.pre_ln(x))
        x = x + self.mlp(self.post_ln(x))
        return x