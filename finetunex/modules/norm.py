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
        x_type = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        return self.weight * norm_x.to(x_type)