"""
RMSNorm
formula: ai' = ai/RMS(a) * gi 
         RMS(a) = sqrt(1/n summation of ai^2)
         (g is a trainable/learnable vector)
"""
import torch
import torch.nn as nn
# TODO: eps value to be used from model config
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps #prevents division by zero
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 'g' learnable vector using Parameter from torch - enable gradient computation during backward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1,keepdim=True).sqrt()
        return self.weight * (x / (rms + self.eps))